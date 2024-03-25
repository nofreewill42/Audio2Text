from torch import nn
import torch.nn.functional as F


class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0)
        self.padding = padding
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x, x_lens):
        x = F.pad(x, self.padding)  # nn.Conv2d does not support asymmetric padding
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x_lens = (x_lens - 1) // 2 + 1
        return x, x_lens

class CNNEmbedder(nn.Module):
    #def __init__(self, d_model, dimensions, kernel_sizes, strides, paddings):
    def __init__(self, config):
        super().__init__()

        # CONFIG - START
        d_model = config["d_model"]
        cnn_embedder_config = config["cnn_embedder"]
        dimensions = cnn_embedder_config["dimensions"]
        kernel_sizes = cnn_embedder_config["kernel_sizes"]
        strides = cnn_embedder_config["strides"]
        paddings = cnn_embedder_config["paddings"]
        # CONFIG - END

        self.convs = nn.ModuleList([
            CNNLayer(dimensions[i], dimensions[i+1], kernel_sizes[i], strides[i], paddings[i])
            for i in range(len(dimensions)-1)
        ])
        self.cnn2linear = nn.Linear(512*16, d_model)
    
    def forward(self, x, x_lens, n_cnn_processed=0):
        # x: (batch, 1, freq, time)
        for conv in self.convs:
            x, x_lens = conv(x, x_lens)
        x = x.flatten(1,2).transpose(1,2)
        x = x[:,n_cnn_processed:]
        x = self.cnn2linear(x)
        return x, x_lens