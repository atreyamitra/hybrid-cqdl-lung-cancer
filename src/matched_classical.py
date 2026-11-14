"""Matched-bottleneck classical comparator.

Identical to the hybrid model except that the eight-qubit quantum layer is
replaced by a small classical transformation of the same eight features:

    conv blocks -> flatten 57,600 -> Linear(57,600 -> 8)      (same as hybrid)
    -> Linear(8 -> 8) -> tanh                                  (replaces quantum layer)
    -> Linear(8 -> 1) -> sigmoid                               (same as hybrid)

tanh keeps the eight intermediate values in [-1, 1], the same range as the
hybrid's Pauli-Z expectation values. The replacement block has 72 trainable
parameters (64 weights + 8 biases) versus 24 rotation angles in the three-layer
circuit, so the two models are matched in bottleneck width and interface, not
exactly in parameter count (479,705 vs 479,657).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchedBottleneckCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 30 * 30, 8)
        self.mixer = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 1)

    def bottleneck(self, x):
        """The eight values that replace the quantum layer's expectation values."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return torch.tanh(self.mixer(self.fc1(torch.flatten(x, 1))))

    def forward(self, x):
        return torch.sigmoid(self.fc2(self.bottleneck(x)))
