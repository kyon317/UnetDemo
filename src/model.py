import torch
import torch.nn as nn
import torch.nn.functional as F


# import any other libraries you need below this line

class twoConvBlock(nn.Module):
    """Part 1  The Convolutional blocks"""

    # initialize the block
    def __init__(self, input_channel, output_channel):
        super(twoConvBlock, self).__init__()
        self.doubleConvBlock = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),  # 3 × 3 un-padded convolution layer
            nn.ReLU(inplace=True),  # ReLU
            nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1),  # 3 × 3 un-padded convolution layer
            nn.BatchNorm2d(output_channel),  # Batch normalization layer
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # implement the forward path
        return self.doubleConvBlock(x)


class downStep(nn.Module):
    """Part 2  The Contracting path"""

    # initialize the down path
    def __init__(self, input_channel, output_channel):
        super(downStep, self).__init__()
        self.convBlock = twoConvBlock(input_channel, output_channel)  # 2 conv blocks
        self.maxPool = nn.MaxPool2d(kernel_size=2)  # 2 x 2 max pool

    def forward(self, x):
        # implement the forward path
        x = self.convBlock(x)
        x_maxPool = self.maxPool(x)
        return x, x_maxPool


class upStep(nn.Module):
    """Part 3  The Expansive path"""

    def __init__(self, input_channel, output_channel):
        super(upStep, self).__init__()
        # initialize the up path
        self.upConv = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2)  # transpose convolutions
        self.convBlock = twoConvBlock(output_channel * 2, output_channel)  #

    def forward(self, x, skip_connection):
        # implement the forward path
        x = self.upConv(x)

        # process crop and copy
        # reference : https://github.com/milesial/Pytorch-UNet/blob/67bf11b4db4c5f2891bd7e8e7f58bcde8ee2d2db/unet/unet_parts.py
        # target_size = (x.size(2), x.size(3))
        # skip_connection = v1.center_crop(skip_connection, output_size=target_size)
        diffY = skip_connection.size()[2] - x.size()[2]
        diffX = skip_connection.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip_connection, x], dim=1)
        new_x = self.convBlock(x)
        return new_x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        # initialize the complete model
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Contracting
        self.inc = downStep(n_channels, 64)
        self.down1 = downStep(64, 128)
        self.down2 = downStep(128, 256)
        self.down3 = downStep(256, 512)

        # Bottom, no max pooling
        self.bot = twoConvBlock(512, 1024)

        # Expansive
        self.up1 = upStep(1024, 512)
        self.up2 = upStep(512, 256)
        self.up3 = upStep(256, 128)
        self.up4 = upStep(128, 64)

        # Output
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # implement the forward path
        x1, x1_maxpool = self.inc(x)
        x2, x2_maxpool = self.down1(x1_maxpool)
        x3, x3_maxpool = self.down2(x2_maxpool)
        x4, x4_maxpool = self.down3(x3_maxpool)

        x_bot = self.bot(x4_maxpool)

        x = self.up1(x_bot, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x_out = self.outc(x)

        return F.sigmoid(x_out)
