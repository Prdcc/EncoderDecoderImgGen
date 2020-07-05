import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, activation = nn.ReLU(), latentSize = 10):
        super(Encoder, self).__init__()
        kwargsConv = {"kernel_size": 5,"stride":1,"dilation":1}

        self.conv1 = nn.Sequential(
            nn.Conv2d(1,6,**kwargsConv),
            activation,
            nn.Conv2d(6,6,3,padding = 1),
            activation,
            nn.MaxPool2d(kernel_size = 2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6,16,**kwargsConv),
            activation,
            nn.Conv2d(16,16,3,padding = 1),
            activation,
            nn.AvgPool2d(kernel_size = 2, stride=2)
        )

        self.layerOut = nn.Sequential(
            nn.Linear(256,120),
            activation,
            nn.Linear(120,84),
            activation,
            nn.Linear(84,latentSize)
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.layerOut(out)
        return out

class Decoder(nn.Module):
    def __init__(self, activation = nn.ReLU(), latentSize = 10):
        super(Decoder, self).__init__()
        kwargsConv = {"kernel_size": 5,"stride":1,"dilation":1}

        self.layerIn = nn.Sequential(
            nn.Linear(latentSize, 96),
            activation,
            nn.Linear(96, 96)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(6,6,kernel_size=3,padding=5),
            activation,
            nn.Conv2d(6,6,kernel_size=3,padding=5),
            activation
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6,6,kernel_size=3,padding=3),
            activation,
            nn.Conv2d(6,6,kernel_size=3,padding=3),
            activation
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(6,6,kernel_size=3,padding=1),
            activation,
            nn.Conv2d(6,1,kernel_size=3,padding=1),
            nn.Tanh
        )

    def forward(self, x):
        out = self.layerIn(x)
        out = out.view(out.size(0),6,4,4)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out
