import math
import torch
import torch.nn as nn
from .nn.conv import Conv, Concat
from .nn.block import SPPF, C2f
from .nn.head import Detect
from .nn.utils import initialize_weights


def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    # import ast

    # Args
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            # LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        # if verbose:
        #     LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    # if verbose:
    #     LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = eval(a)  # eval strings
                except NameError:
                    args[j] = a
                # with contextlib.suppress(ValueError):
                #     args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (
            # Classify,
            Conv,
            # ConvTranspose,
            # GhostConv,
            # Bottleneck,
            # GhostBottleneck,
            # SPP,
            SPPF,
            # DWConv,
            # Focus,
            # BottleneckCSP,
            # C1,
            # C2,
            C2f,
            # C3,
            # C3TR,
            # C3Ghost,
            # nn.ConvTranspose2d,
            # DWConvTranspose2d,
            # C3x,
            # RepC3,
        ):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (C2f,):
                args.insert(2, n)  # number of repeats
                n = 1
        # elif m is AIFI:
        #     args = [ch[f], *args]
        # elif m in (HGStem, HGBlock):
        #     c1, cm, c2 = ch[f], args[0], args[1]
        #     args = [c1, cm, c2, *args[2:]]
        #     if m is HGBlock:
        #         args.insert(4, n)  # number of repeats
        #         n = 1
        # elif m is ResNetLayer:
        #     c2 = args[1] if args[3] else args[1] * 4
        # elif m is nn.BatchNorm2d:
        #     args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (Detect,):
            args.append([ch[x] for x in f])
        #     if m is Segment:
        #         args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        # elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
        #     args.insert(1, [ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        # if verbose:
        #     LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class DetectionModel(nn.Module):
    def __init__(self, cfg: dict, ch: int = None, nc: int = None, names: dict = None):
        super().__init__()
        self.cfg = cfg
        self.ch = ch or cfg.get("ch", 3)
        self.nc = nc or cfg.get("nc", 80)
        self.names = names or cfg.get("names", None) or {i: f"{i}" for i in range(self.nc)}
        self.inplace = cfg.get("inplace", True)

        self.model, self.save = parse_model(self.cfg, self.ch)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect,)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # forward = self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, self.ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once

        # Init weights, biases
        initialize_weights(self.model)

    def forward(self, x):
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # if profile:
            #     self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            # if visualize:
            #     feature_visualization(x, m.type, m.i, save_dir=visualize)
            # if embed and m.i in embed:
            #     embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
            #     if m.i == max(embed):
            #         return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x