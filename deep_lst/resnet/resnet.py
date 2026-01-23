import torch
import torch.nn as nn
import torch.nn.functional as F
import code


class BasicBlock(nn.Module):
    """Minimal ResNet basic block: Conv-BN-ReLU-Conv-BN + skip."""

    def __init__(self, in_size: int, out_size: int, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class TinyResNet(nn.Module):
    """A very small ResNet."""
    def __init__(self, in_size, out_size, **kwargs):
        super().__init__()

        hidden_sizes = kwargs['hidden_sizes']

        self.stem = nn.Sequential(
            nn.Conv2d(in_size, hidden_sizes[0], kernel_size=kwargs['kernel_size'], padding=1, bias=False),
            nn.BatchNorm2d(hidden_sizes[0]),
            nn.ReLU(inplace=True),
        )

        self.layers = []
        for h in range(len(hidden_sizes) - 1):
            self.layers.append(self._make_layer(hidden_sizes[h], hidden_sizes[h+1], n_blocks=kwargs['n_blocks'], kernel_size=kwargs['kernel_size']))

        self.head = nn.Conv2d(hidden_sizes[-1], out_size, kernel_size=1)

    def _make_layer(self, in_size: int, out_size: int, kernel_size: int, n_blocks: int) -> nn.Sequential:
        blocks = [BasicBlock(in_size, out_size, kernel_size)]
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock(out_size, out_size, kernel_size))
        return nn.Sequential(*blocks)

    def forward(self, *inputs) -> torch.Tensor:

        inputs = [x for x in inputs if x is not None]

        x = torch.cat(inputs, dim=2) # assumes (B, T, C, H, W)
        x = x[:,0,...]  # remove time dim

        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        x = self.head(x)
        x.unsqueeze_(1)  # add time dim back
        return x


