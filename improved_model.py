"""
ImprovedNet: a compact CIFAR-style ResNet for 32x32 MiniPlaces scene recognition.

This is an additive extension to the assignment baseline (student_code.LeNet).
It is NOT part of the graded submission; it exists to push validation top-1
accuracy well above the LeNet-5 baseline (~19.4%).

Design notes:
  * 3x3 stem with NO early downsampling (ImageNet-style 7x7 + maxpool would
    destroy a 32x32 image). This follows the CIFAR ResNet variant (He et al.).
  * Three residual stages with widths 64 -> 128 -> 256 and stride-2 downsampling
    between stages, ending at an 8x8 feature map.
  * Global average pooling + dropout + a single linear classifier.
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """A standard ResNet basic block: two 3x3 convs with a residual shortcut."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # conv bias is disabled because the following BatchNorm absorbs the bias.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut: only needed when shape changes (stride or width).
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # residual connection
        return torch.relu(out)


class ImprovedNet(nn.Module):
    """Compact CIFAR-style ResNet (~2.8M params) for 100-class 32x32 inputs."""

    def __init__(self, num_classes=100, blocks_per_stage=2, dropout=0.3):
        super().__init__()
        self.in_channels = 64

        # Stem keeps full 32x32 resolution.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Three residual stages; stride 2 halves the spatial size at each new stage.
        self.layer1 = self._make_stage(64, blocks_per_stage, stride=1)   # 32x32
        self.layer2 = self._make_stage(128, blocks_per_stage, stride=2)  # 16x16
        self.layer3 = self._make_stage(256, blocks_per_stage, stride=2)  # 8x8

        self.pool = nn.AdaptiveAvgPool2d(1)   # global average pooling -> [N, 256, 1, 1]
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

        self._init_weights()

    def _make_stage(self, out_channels, n_blocks, stride):
        # The first block in a stage may downsample; the rest keep the shape.
        strides = [stride] + [1] * (n_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)

    def _init_weights(self):
        # Kaiming init for convs, standard init for BN — a common, robust recipe.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def count_model_params(model):
    """Return the number of trainable parameters (raw count, not in millions)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Sanity check: parameter count and a dummy forward pass.
    model = ImprovedNet()
    n_params = count_model_params(model)
    print(f"ImprovedNet trainable params: {n_params:,} ({n_params / 1e6:.3f}M)")

    dummy = torch.randn(2, 3, 32, 32)
    out = model(dummy)
    print(f"Input  shape: {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}  (expected (2, 100))")
    assert out.shape == (2, 100), "Unexpected output shape!"
    print("Forward pass OK.")
