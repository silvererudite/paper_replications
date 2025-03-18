import torch
import torch.nn as nn
from custom_gelu import CustomGelu


class CustomPositionWiseFFN(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()
        self.gelu = CustomGelu()

    def forward(self, in_features, weights):
        linear_transform_one = self.gelu(in_features @ weights["w1.weight"].T)
        linear_transform_two = linear_transform_one @ weights["w2.weight"].T
        return linear_transform_two
