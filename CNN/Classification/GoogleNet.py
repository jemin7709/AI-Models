import torch
import torch.nn as nn

class CBRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.Block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        x = self.Block(x)
        return x

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, in_3x3, out_3x3, in_5x5, out_5x5, out_pool):
        super().__init__()
        self.branch1 = CBRBlock(in_channels, out_1x1, kernel_size=(1, 1), stride=(1, 1))
        self.branch2 = nn.Sequential(
            CBRBlock(in_channels, in_3x3, kernel_size=(1, 1)),
            CBRBlock(in_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1))
        )
        self.branch3 = nn.Sequential(
            CBRBlock(in_channels, in_5x5, kernel_size=(1, 1)),
            CBRBlock(in_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2))
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            CBRBlock(in_channels, out_pool, kernel_size=(1, 1))
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.AuxBlock1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3)),
            CBRBlock(in_channels, 128, kernel_size=(1, 1))
        )
        self.AuxBlock2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(p=0.7),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.AuxBlock1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.AuxBlock2(x)
        return x

class GoogleNet(nn.Module):
    def __init__(self, aux_logits=True, num_classes=100):
        super().__init__()
        self.aux_logits = aux_logits
        
        self.conv1 = CBRBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = CBRBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)

        if self.aux_logits and self.training:
            return aux1, aux2, x
        else:
            return x

def test():
    model = GoogleNet()
    aux1, aux2, output = model(torch.randn(128, 3, 224, 224)).to("cuda")
    print(output.size())

# test()