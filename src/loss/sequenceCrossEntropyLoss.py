import torch
from torch import nn
import torch.nn.functional as F

def to_contiguous(tensor):
    return tensor if tensor.is_contiguous() else tensor.contiguous()


class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self, sequence_normalize=False, sample_normalize=True):
        super(SequenceCrossEntropyLoss, self).__init__()

        # Initialize parameters
        self.sequence_normalize = sequence_normalize
        self.sample_normalize = sample_normalize

        assert not (sequence_normalize and sample_normalize), "Both sequence_normalize and sample_normalize cannot be True."

    def forward(self, input, target, length):
        # Create a binary mask to cover valid positions in each sequence
        mask = torch.zeros_like(target)
        for i in range(target.size(0)):
            mask[i, :length[i]].fill_(1)
        
        # Ensure input and target have the same sequence length
        max_length = max(length)
        target = target[:, :max_length]
        mask = mask[:, :max_length]
        
        # Reshape input tensor and apply log softmax
        input = to_contiguous(input).view(-1, input.size(2))
        input = F.log_softmax(input, dim=1)
        
        # Reshape target and mask tensors
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        
        # Compute the negative log likelihood loss
        output = -input.gather(1, target.long()) * mask
        output = torch.sum(output)
        
        # Normalize the loss if specified
        if self.sequence_normalize:
            output = output / torch.sum(mask)
        if self.sample_normalize:
            output = output / target.size(0)
        
        return output