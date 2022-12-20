import torch
import torch.nn as nn

config = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGGNet(nn.Module):
    
    def __init__(self, type='D', num_classes=100):
        """
        A: VGG11, B: VGG13, D: VGG16, E: VGG19
        """
        super().__init__()
        self.feature = self.make_layer(config[type])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes)
        )
    
    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def make_layer(self, config):
        layers = []
        in_channels = 3
        for layer in config:
            if layer == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.append(nn.Conv2d(in_channels, layer, kernel_size=3, padding=1))
                in_channels = layer
        return nn.Sequential(*layers)

def test():
    model = VGGNet()
    output = model(torch.randn(128, 3, 224, 224)).to("cuda")
    print(output.size())

# test()