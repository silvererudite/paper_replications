import torch
import torch.nn as nn
from custom_softmax import CustomSoftmax
class CustomScaledDotProductAttention(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()

    def forward(self,
                K: torch.FloatTensor,
                Q: torch.FloatTensor,
                V: torch.FloatTensor,
                mask: torch.BoolTensor = None,
                pdrop: float = None
                ):
        d_k = Q.shape[-1]
        custom_softmax = CustomSoftmax()
        pre_soft_max = (Q @ K.transpose(-2, -1)) / (torch.sqrt(torch.tensor(d_k)))
        new_mask = torch.where(mask, torch.tensor(float('-inf')), 0) if mask is not None else 0
        pre_soft_max = pre_soft_max + new_mask
        attn_weights = custom_softmax(pre_soft_max, -1) @ V
        if not pdrop:
            return attn_weights
        keep_prob = 1 - pdrop
        mask = (torch.rand_like(attn_weights) < keep_prob).float()

        return attn_weights * mask / keep_prob