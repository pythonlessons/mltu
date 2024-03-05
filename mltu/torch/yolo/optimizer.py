import torch
from torch import nn

class AccumulativeOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer, batch_size, nbs=64):
        super(AccumulativeOptimizer, self).__init__(optimizer.param_groups, optimizer.defaults)
        self.optimizer = optimizer
        self.accumulation_steps = int(nbs / batch_size)
        self.current_step = 0

    def zero_grad(self):
        if self.current_step == 0:
            self.optimizer.zero_grad()

    def step(self):
        self.current_step += 1
        if self.current_step >= self.accumulation_steps:
            self.optimizer.step()
            self.current_step = 0
            self.optimizer.zero_grad()


def build_optimizer(model, name: str="AdamW", lr: float=1e-3, weight_decay: float=0.0, momentum: float=0.937, decay=0.0005):

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in fullname:  # bias (no decay)
                pg2.append(param)
            elif isinstance(module, bn):  # weight (no decay)
                pg1.append(param)
            else:  # weight (with decay)
                pg0.append(param)

    if name == "AdamW":
        optimizer = torch.optim.AdamW(pg2, lr=lr, weight_decay=weight_decay, betas=(momentum, 0.999))
    elif name == "Adam":
        optimizer = torch.optim.Adam(pg2, lr=lr, weight_decay=weight_decay, betas=(momentum, 0.999))
    elif name == "SGD":
        optimizer = torch.optim.SGD(pg2, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {name} not supported!")
    
    optimizer.add_param_group({'params': pg0, 'weight_decay': decay})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg1, 'weight_decay': 0.0})  # add pg2 (biases)

    del pg0, pg1, pg2

    return optimizer