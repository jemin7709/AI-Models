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

"""
phi, resolution, drop_rate
"""
phi_value = {
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5)
}
class CBSBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride, padding, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x)

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

class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride, padding, expand_ratio, stochastic=0.8, reduction=4):
        super().__init__()
        hidden_ch = in_ch * expand_ratio
        reduction_ch = int(in_ch / reduction)
        self.stochastic = stochastic
        self.use_skipconnection = in_ch == out_ch and stride == 1
        self.expand_condition = in_ch != hidden_ch

        if self.expand_condition:
            self.expand = CBSBlock(in_ch, hidden_ch, 3, 1, 'same')
        self.depthwise = CBSBlock(hidden_ch, hidden_ch, k_size, stride, padding, groups=hidden_ch)
        self.seblock = SEBlock(hidden_ch, reduction_ch)
        self.output = nn.Sequential(
            nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
    
    def stochastic_depth(self, x):
        if not self.training:
            return x
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.stochastic
        return torch.div(x, self.stochastic) * binary_tensor

    def forward(self, input):
        x = self.expand(input) if self.expand_condition else input
        if self.use_skipconnection:
            x = self.depthwise(x)
            x = self.seblock(x)
            x = self.output(x)
            x = self.stochastic_depth(x)
            return x + input
        else:
            x = self.depthwise(x)
            x = self.seblock(x)
            x = self.output(x)
            return x

class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super().__init__()
        width, depth, drop = self.calculate_factor(version)
        last_ch = math.ceil(1280 * width)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width, depth, last_ch)
        self.classifier = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(last_ch, num_classes)
        )

    def calculate_factor(self, version, alpha=1.2, beta=1.1):
        phi, _, drop = phi_value[version]
        depth = alpha ** phi
        width = beta ** phi
        return width, depth, drop
    
    def create_features(self, width, depth, last_ch):
        channels = int(32 * width)
        features = [CBSBlock(3, channels, 3, stride=2, padding=1)]
        in_ch = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_ch = 4 * math.ceil(int(channels * width) / 4)
            repeat = math.ceil(repeats * depth)

            for layer in range(repeat):
                features.append(
                    MBConv(
                        in_ch, 
                        out_ch,
                        kernel_size,
                        stride=stride if layer == 0 else 1,
                        padding=kernel_size//2,
                        expand_ratio=expand_ratio
                    )
                )
                in_ch = out_ch

        features.append(CBSBlock(in_ch, last_ch, k_size=1, stride=1, padding=0))
        return nn.Sequential(*features)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))

def test():
    model = EfficientNet(10)
    output = model(torch.randn(128, 3, 224, 224))
    print(output.size())

# test()

    