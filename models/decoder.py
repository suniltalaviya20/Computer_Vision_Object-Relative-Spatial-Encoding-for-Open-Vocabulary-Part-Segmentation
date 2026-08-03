from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBlock(nn.Module):
    """Two convolution layers with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class TinyPartDecoder(nn.Module):
    """Small encoder-decoder for debugging the training pipeline.

    Input channels:

    - RGB image: 3
    - parent-object mask: 1
    - relative U coordinate: 1
    - relative V coordinate: 1

    Total: 6 channels
    """

    def __init__(
        self,
        input_channels: int = 6,
    ) -> None:
        super().__init__()

        self.encoder_1 = ConvBlock(
            input_channels,
            32,
        )

        self.pool_1 = nn.MaxPool2d(2)

        self.encoder_2 = ConvBlock(
            32,
            64,
        )

        self.pool_2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(
            64,
            128,
        )

        self.up_2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2,
        )

        self.decoder_2 = ConvBlock(
            64 + 64,
            64,
        )

        self.up_1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2,
        )

        self.decoder_1 = ConvBlock(
            32 + 32,
            32,
        )

        self.output_layer = nn.Conv2d(
            32,
            1,
            kernel_size=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        encoder_1 = self.encoder_1(x)

        encoder_2 = self.encoder_2(
            self.pool_1(encoder_1)
        )

        bottleneck = self.bottleneck(
            self.pool_2(encoder_2)
        )

        decoder_2 = self.up_2(bottleneck)

        decoder_2 = torch.cat(
            [decoder_2, encoder_2],
            dim=1,
        )

        decoder_2 = self.decoder_2(
            decoder_2
        )

        decoder_1 = self.up_1(decoder_2)

        decoder_1 = torch.cat(
            [decoder_1, encoder_1],
            dim=1,
        )

        decoder_1 = self.decoder_1(
            decoder_1
        )

        return self.output_layer(decoder_1)