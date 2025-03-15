import torch
import torch.nn as nn


# for a pre-norm transformer block
class CustomRMSNorm(nn.Module):
    def __init__(
            self,
            eps: float
    ):
        super().__init__()
        self.eps = eps

    def forward(self, d_model, weights, in_features):
        rms = torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + self.eps)
        normalized_features = in_features / rms
        return normalized_features * weights["weight"]
