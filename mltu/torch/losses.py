import torch
import torch.nn as nn

class CTCLoss(nn.Module):
    """ CTC loss for PyTorch
    """
    def __init__(self, blank: int):
        """ CTC loss for PyTorch

        Args:
            blank: Index of the blank label
        """
        super(CTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank)

    def forward(self, output, target):
        """
        Args:
            output: Tensor of shape (batch_size, num_classes, sequence_length)
            target: Tensor of shape (batch_size, sequence_length)
            
        Returns:
            loss: Scalar
        """
        output = output.permute(1, 0, 2)  # (sequence_length, batch_size, num_classes)
        output_lengths = torch.full(size=(output.size(1),), fill_value=output.size(0), dtype=torch.int32)
        target_lengths = torch.full(size=(output.size(1),), fill_value=target.size(1), dtype=torch.int32)

        loss = self.ctc_loss(output, target, output_lengths, target_lengths)

        return loss