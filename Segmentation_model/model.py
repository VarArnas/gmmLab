import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, input_c, output_c):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_c, output_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_c, output_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
    
class SegmentationModel(nn.Module):
    def __init__(
            self, input_c=3, output_c = 4, features = [64, 128, 256, 512]
    ):
        super(SegmentationModel, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropoutEarly = nn.Dropout2d(p=0.3)
        self.dropoutLate = nn.Dropout2d(p=0.5)

        for feature in features:
            self.downs.append(DoubleConv(input_c, feature))
            input_c = feature
        
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.finalConv = nn.Conv2d(features[0], output_c, 1)

    def forward(self, x):
        skip_connections = []

        for i, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            if i > 1:
                x = self.dropoutEarly(x)

        x = self.bottleneck(x)
        x = self.dropoutLate(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            concat = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat)

        return self.finalConv(x)