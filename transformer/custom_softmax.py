import torch
import torch.nn as nn
class CustomSoftmax(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()

    def forward(self, in_features, ith_dim):
        max_val, _ = torch.max(in_features, dim=ith_dim, keepdim=True)
        safe_input = in_features - max_val
        exp_val = torch.exp(safe_input)
        softmaxxed_thing = exp_val / torch.sum(exp_val, dim=ith_dim, keepdim=True)
        return softmaxxed_thing