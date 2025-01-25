from typing import List

import torch
from torch import nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class ConvBlock(nn.Module):
    def __init__(self, in_num_channels: int, out_num_channels: int, *args, **kwargs):
        super().__init__()
        self._layers = nn.Sequential(
            nn.Conv2d(in_num_channels, out_num_channels, *args, **kwargs, bias=False),
            nn.BatchNorm2d(out_num_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_num_channels: int, out_num_channels: int, upsampling_scale: int = 2):
        super().__init__()
        self._upsample = nn.Upsample(scale_factor=upsampling_scale)
        self._conv_blocks = nn.Sequential(
            ConvBlock(in_num_channels, out_num_channels, kernel_size=3, padding=1),
            ConvBlock(out_num_channels, out_num_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        x = self._upsample(x)
        x = torch.cat([x, skip_connection], dim=1)  # Alongside channel dimension
        return self._conv_blocks(x)


class Decoder(nn.Module):
    def __init__(self, in_num_chs: int, skips_num_chs: List[int], decoder_num_chs: List[int]):
        """Pass channels nums from the inner-most to the outer-most layer of the UNet."""
        super().__init__()
        in_num_chs = [in_num_chs, *decoder_num_chs[:-1]]
        self.blocks = nn.ModuleList()
        for in_ch, skip_ch, out_ch in zip(in_num_chs, skips_num_chs, decoder_num_chs):
            self.blocks.append(DecoderBlock(in_ch + skip_ch, out_ch))

    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            x = block(x, skip_connections[i])
        return x


class Unet(nn.Module):
    # First conv in torchvision resnet is slightly different from typical unet encoder
    IN_NUM_CHS = 3
    RESNET_18_BLOCKS_NAMES = ["relu", "layer1", "layer2", "layer3", "layer4"]
    RESNET_18_BLOCKS_NUM_CHS = [64, 64, 128, 256, 512]
    DECODER_BLOCKS_NUM_CHS = [512, 256, 128, 64, 32]
    SKIPS_NUM_CHS = [IN_NUM_CHS, *RESNET_18_BLOCKS_NUM_CHS[:-1]]

    def __init__(self, num_classes: int):
        super().__init__()
        self.encoder = models.resnet18()
        self.encoder_features = create_feature_extractor(self.encoder, self.RESNET_18_BLOCKS_NAMES)
        self.decoder = Decoder(
            in_num_chs=self.RESNET_18_BLOCKS_NUM_CHS[-1],
            skips_num_chs=list(reversed(self.SKIPS_NUM_CHS)),
            decoder_num_chs=self.DECODER_BLOCKS_NUM_CHS,
        )
        self.segmentation_head = nn.Conv2d(self.DECODER_BLOCKS_NUM_CHS[-1], num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_features = list(self.encoder_features(x).values())
        encoder_features.reverse()
        encoded, *skip_connections = encoder_features
        decoded = self.decoder(encoded, [*skip_connections, x])
        return self.softmax(self.segmentation_head(decoded))
