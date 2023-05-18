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
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=False)
        self.blank = blank

    def forward(self, output, target):
        """
        Args:
            output: Tensor of shape (batch_size, num_classes, sequence_length)
            target: Tensor of shape (batch_size, sequence_length)
            
        Returns:
            loss: Scalar
        """
        # Remove padding and blank tokens from target
        target_lengths = torch.sum(target != self.blank, dim=1)
        target_unpadded = target[target != self.blank].view(-1)

        output = output.permute(1, 0, 2)  # (sequence_length, batch_size, num_classes)
        output_lengths = torch.full(size=(output.size(1),), fill_value=output.size(0), dtype=torch.int64)

        loss = self.ctc_loss(output, target_unpadded, output_lengths, target_lengths)

        return loss