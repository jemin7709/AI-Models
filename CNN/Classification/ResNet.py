import torch
import torch.nn as nn

class BottleNeck(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1) -> None:
        super().__init__()
        self.expension = 4
        self.Block1 = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        )
        self.Block2 = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        )
        self.Block3 = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch * self.expension, kernel_size=1, bias=False)
        )
        if stride != 1 or in_ch != out_ch * self.expension:
            self.identity = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch, out_ch * self.expension, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.identity = None

    def forward(self, x):
        identity = x
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        if self.identity is not None:
            identity = self.identity(identity)
        x += identity
        return x

class ResNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, layers=[3, 4, 6, 3], num_classes=100):
        """
        ResNet50: [3, 4, 6, 3],
        ResNet101: [3, 4, 23, 3],
        ResNet152: [3, 8, 36, 3],
        """
        super().__init__()
        self.in_ch = out_ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self.make_layer(64, layers[0], 1)
        self.conv3 = self.make_layer(128, layers[1], 2)
        self.conv4 = self.make_layer(256, layers[2], 2)
        self.conv5 = self.make_layer(512, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)
        
    def make_layer(self, out_ch, iter, stride):
        layers = []
        layers.append(BottleNeck(self.in_ch, out_ch, stride))
        self.in_ch = out_ch * 4
        for _ in range(iter - 1):
            layers.append(BottleNeck(self.in_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def test():
    model = ResNet()
    output = model(torch.randn(128, 3, 224, 224)).to("cuda")
    print(output.size())

# test()