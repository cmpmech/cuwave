"""Neural networks as reparametrization of design field (using PyTorch)
see https://www.sciencedirect.com/science/article/pii/S0045782523004024
"""

import torch
from torch import nn

CONVOLUTIONS = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
POOLINGS = {1: nn.AvgPool1d, 2: nn.AvgPool2d, 3: nn.AvgPool3d}
OUTPUT_STD = 0.01


# -------------------------------------- helpers --------------------------------------
def _initialize(model: nn.Module) -> None:
    """Xavier normal with zero bias."""
    for module in model.modules():
        if isinstance(module, tuple(CONVOLUTIONS.values())):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)


def nn_params(model: nn.Module) -> int:
    """Number of trainable parameters in `model`"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _convolution(
    in_channels: int, out_channels: int, kernel: int, dim: int
) -> nn.Module:
    """`kernel`-wide convolution over `dim` axes, padded to retain the resolution"""
    if dim not in CONVOLUTIONS:
        raise ValueError(f"dim is 1, 2 or 3, not {dim!r}")
    return CONVOLUTIONS[dim](in_channels, out_channels, kernel, padding=kernel // 2)


def _block(
    in_channels: int,
    out_channels: int,
    kernel: int,
    activation: type[nn.Module],
    dim: int,
) -> nn.Sequential:
    """convolve, normalize, activate: the unit a stack is built from

    `GroupNorm(1, channels)` normalizes each sample over channels and space (layer
    normalization).
    """
    return nn.Sequential(
        _convolution(in_channels, out_channels, kernel, dim),
        nn.GroupNorm(1, out_channels),
        activation(),
    )


# -------------------------------------- networks -------------------------------------
class Generator(nn.Module):
    """Fixed noise to field, doubling the resolution at every channel transition.

    Nearest-neighbour upsampling per transition, and the stack ends in a sigmoid, so the
    field is in [0, 1]. The output convolution starts with small random weights and
    `output_bias`, so the field starts flat at 1, where the pixel-wise drivers start.

    Args:
        channels: latent channels, tapering to the one channel of the field.
        shape: field resolution, divisible by 2**(len(channels) - 1).
        kernel: convolution width, padded to retain the resolution.
        activation: hidden activation; the output activation is always a sigmoid.
        dim: 1, 2 or 3.
        output_bias: bias into the output sigmoid, so a large value starts the field
            flat at 1.
        learnable: make `latent` a parameter instead of a buffer.
    """

    def __init__(
        self,
        channels: list[int],
        shape: tuple[int, ...],
        kernel: int = 5,
        activation: type[nn.Module] = nn.GELU,
        dim: int = 2,
        output_bias: float = 10.0,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        if len(shape) != dim:
            raise ValueError(f"a {dim}D generator needs {dim} sizes, got {shape}")
        blocks = len(channels) - 1
        step = 2**blocks
        if any(n % step for n in shape):
            raise ValueError(f"{blocks} upsamplings need {shape} divisible by {step}")

        layers = []
        for i in range(blocks):
            last = i == blocks - 1
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.GroupNorm(1, channels[i]),
                _convolution(channels[i], channels[i + 1], kernel, dim),
                nn.Sigmoid() if last else activation(),
            ]
        self.stack = nn.Sequential(*layers)
        _initialize(self)

        output = self.stack[-2]  # the convolution the sigmoid reads
        nn.init.normal_(output.weight, std=OUTPUT_STD)
        nn.init.constant_(output.bias, output_bias)

        noise = torch.randn(1, channels[0], *(n // step for n in shape))
        noise = 2.0 * noise / (noise.max() - noise.min())  # span activation's range
        if learnable:
            self.latent = nn.Parameter(noise)
        else:
            self.register_buffer("latent", noise)

    def forward(self) -> torch.Tensor:
        return self.stack(self.latent)
