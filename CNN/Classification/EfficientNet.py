import torch
import torch.nn as nn
import math

"""
expend ratio, channels, layers, stride(downsample), kernel size
"""
base_model = [
    [1,  16, 1, 1, 3],
    [6,  24, 2, 2, 3],
    [6,  40, 2, 2, 5],
    [6,  80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3]
]

class SEBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.SiLU(),
            nn.Linear(out_ch, in_ch),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        squeeze = self.squeeze(x)
        squeeze = torch.flatten(squeeze, 1)
        excitation = self.excitation(squeeze).view(x.shape[0], x.shape[1], 1, 1)
        return x * excitation

    

def test():
    model = SEBlock(3, 3)
    output = model(torch.randn(128, 3, 224, 224))
    print(output.size())

test()

    