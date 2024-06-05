import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Modified Attention Unet:
We use atrous spatial pyramid pooling, with encoder, decoder structure
in conjunction with applying attention scores
"""

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, dilation):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.final_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.relu(self.bn(self.conv1(x)))
        x2 = self.relu(self.bn(self.conv2(x)))
        x3 = self.relu(self.bn(self.conv3(x)))
        x4 = self.relu(self.bn(self.conv4(x)))
        
        x5 = self.pool(x)
        x5 = self.conv_pool(x5)
        x5 = self.relu(x5)
        
        # Upsample the pooled feature map to the same spatial size as x1, x2, x3, x4
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        # Concatenate along the channel dimension
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
        x = self.final_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x

class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out

class ModifiedAttUNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dilations=[4, 3, 2, 1], features=[64, 128, 256, 512]):
        super(ModifiedAttUNET, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for dilation, feature in zip(dilations, features):
            self.encoder.append(DoubleConv(in_channels, feature, dilation))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dilation=1)

        self.aspp = ASPP(features[-1] * 2, features[-1] * 2)

        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature, dilation=1))
            self.decoder.append(
                AttentionBlock(F_g=feature, F_l=feature, n_coefficients=feature // 2)
            )

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        x = self.aspp(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 3):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 3]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            attn_block = self.decoder[idx + 2]
            skip_connection = attn_block(gate=x, skip_connection=skip_connection)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)

        return self.final_conv(x)

def test():
    model = ModifiedAttUNET()
    input_tensor = torch.randn((1, 3, 160, 160))
    output = model(input_tensor)
    assert output.shape == (1, 1, 160, 160), f'Wrong output shape: {output.shape}'
    print(output.shape)
if __name__ == "__main__":
    test()
