import torch
import torch.nn as nn
import torch.nn.functional as F

def gn(num_channels, groups=8):
    # keep groups ≤ channels
    return nn.GroupNorm(num_groups=min(groups, num_channels), num_channels=num_channels, affine=True)

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = gn(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = gn(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                gn(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Encoder3D(nn.Module):
    def __init__(self, in_channels, base_channels=8):
        super(Encoder3D, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            gn(base_channels),          # was InstanceNorm3d
            nn.ReLU(inplace=True)
        )
        self.layer1 = BasicBlock3D(base_channels, base_channels * 2, stride=2)
        self.layer2 = BasicBlock3D(base_channels * 2, base_channels * 4, stride=2)
        self.layer3 = BasicBlock3D(base_channels * 4, base_channels * 8, stride=2)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x0, x1, x2, x3


class Decoder3D(nn.Module):
    def __init__(self, base_channels=8):
        super(Decoder3D, self).__init__()

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(base_channels * 16, base_channels * 8, kernel_size=3, padding=1, bias=False),
            gn(base_channels * 8),      # was InstanceNorm3d
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(base_channels * 16, base_channels * 4, kernel_size=3, padding=1, bias=False),
            gn(base_channels * 4),      # was InstanceNorm3d
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(base_channels * 8, base_channels * 2, kernel_size=3, padding=1, bias=False),
            gn(base_channels * 2),      # was InstanceNorm3d
            nn.ReLU()
        )
        self.up4 = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels, kernel_size=3, padding=1, bias=False),
            gn(base_channels),           # was InstanceNorm3d
            nn.ReLU()
        )
        self.final = nn.Conv3d(base_channels, 3, kernel_size=1)  # keep raw scale (no norm/activation)

    def forward(self, x, skips):
        x2, x1, x0 = skips

        x = self.up1(x)
        x = torch.cat([x, x2], dim=1)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)

        x = self.up3(x)
        x = torch.cat([x, x0], dim=1)
        
        x = self.up4(x)
        x = self.final(x)
        return x


class DualBranchResNet3D(nn.Module):
    def __init__(self, base_channels=8):
        super(DualBranchResNet3D, self).__init__()
        self.encoder_dadt = Encoder3D(in_channels=3, base_channels=base_channels)   # dA/dt (3)
        self.encoder_cond = Encoder3D(in_channels=1, base_channels=base_channels)   # cond (1)
        self.decoder = Decoder3D(base_channels=base_channels)

    def forward(self, dadt, cond):
        x0_d, x1_d, x2_d, x3_d = self.encoder_dadt(dadt)
        x0_c, x1_c, x2_c, x3_c = self.encoder_cond(cond)

        x = torch.cat((x3_d, x3_c), dim=1)
        skip2 = torch.cat((x2_d, x2_c), dim=1)
        skip1 = torch.cat((x1_d, x1_c), dim=1)
        skip0 = torch.cat((x0_d, x0_c), dim=1)

        out = self.decoder(x, skips=(skip2, skip1, skip0))
        return out
