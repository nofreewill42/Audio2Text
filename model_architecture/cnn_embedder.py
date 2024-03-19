from torch import nn
import torch.nn.functional as F


class CNNEmbedder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        def layer(in_channels, out_channels, s=2):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, s, 0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),)
        self.conv1 = layer(1, 128)
        self.conv2 = layer(128, 256)
        self.conv3 = layer(256, 512)
        self.cnn2linear = nn.Linear(512*16, d_model)
    
    def forward(self, x, x_lens):
        # x: (batch, 1, freq, time)
        # pad 2 on left side of time
        # x: (batch, 1, freq, 2+time+0)
        x = F.pad(x, (2,0,1,1))
        x = self.conv1(x)
        x = F.pad(x, (2,0,1,1))
        x = self.conv2(x)
        x = F.pad(x, (2,0,1,1))
        x = self.conv3(x)
        x_lens = (x_lens - 1) // 2 + 1
        x_lens = (x_lens - 1) // 2 + 1
        x_lens = (x_lens - 1) // 2 + 1
        x = x.flatten(1,2).transpose(1,2)
        x = self.cnn2linear(x)
        return x, x_lens