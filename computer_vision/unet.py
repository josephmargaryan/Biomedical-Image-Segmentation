import torch
import torch.nn as nn 

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.encoder.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature
        self.bottleneck = DoubleConv(in_channels=features[-1], out_channels=features[-1]*2)
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(in_channels=feature*2, out_channels=feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(in_channels=feature*2, out_channels=feature))
        self.last_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)
    def forward(self, x):
        skip_connections = []
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx//2]
            concat_skip = torch.concat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip)
        return self.last_conv(x)

def test():
    x = torch.randn(16, 3, 256, 256)
    model = UNET()
    preds = model(x)
    print(x.shape)
    print(preds.shape)
if __name__ == "__main__":
    test()
