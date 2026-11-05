"""EdgeNet: a small CNN sized for a microcontroller- or FPGA-class target.

Five 3x3 convolutions (32-32-64-64-128 channels), two max-pools, global average
pooling and one linear layer: about 140 k parameters and 11 M multiply-
accumulates for a 20 x 20 RGB chip. The 128-dimensional pooled vector is the
feature space the monitor watches.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=False),
        )


class EdgeNet(nn.Module):
    def __init__(self, num_classes: int = 2, widths=(32, 64, 128), dropout: float = 0.2):
        super().__init__()
        c1, c2, c3 = widths
        self.body = nn.Sequential(
            ConvBNReLU(3, c1),
            ConvBNReLU(c1, c1),
            nn.MaxPool2d(2),
            ConvBNReLU(c1, c2),
            ConvBNReLU(c2, c2),
            nn.MaxPool2d(2),
            ConvBNReLU(c2, c3),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c3, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(self.body(x)), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.features(x)))


class ExportNet(nn.Module):
    """Inference graph with batch-norm folded into the convolutions.

    Returns (logits, features) so that the monitor can read the feature vector
    without a second pass.
    """

    def __init__(self, net: EdgeNet):
        super().__init__()
        layers = []
        for m in net.body:
            if isinstance(m, ConvBNReLU):
                conv, bn = m[0], m[1]
                fused = torch.nn.utils.fusion.fuse_conv_bn_eval(conv.eval(), bn.eval())
                layers += [fused, nn.ReLU()]
            else:
                layers.append(m)
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = net.fc

    def forward(self, x: torch.Tensor):
        f = torch.flatten(self.pool(self.body(x)), 1)
        return self.fc(f), f


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
