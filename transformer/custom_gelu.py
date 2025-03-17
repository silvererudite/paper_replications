import torch
import torch.nn as nn
class CustomGelu(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()

    def forward(self, in_features):
        return (1 + torch.erf(in_features / torch.sqrt(torch.tensor(2)))) * in_features / 2