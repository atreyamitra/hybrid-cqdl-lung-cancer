import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantum_layer import QuantumLayer, DEFAULT_LAYERS


class HybridModel(nn.Module):
    """CNN feature extractor -> Linear(57,600 -> 8) -> 8-qubit variational layer
    -> Linear(8, 1) -> sigmoid. ``n_layers`` sets the BasicEntanglerLayers depth
    (default 3, the original configuration)."""

    def __init__(self, n_layers=DEFAULT_LAYERS):
        super(HybridModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)

        # Output 8 features for 8 qubits
        self.fc1 = nn.Linear(64 * 30 * 30, 8)
        self.quantum = QuantumLayer(n_layers)
        self.fc2 = nn.Linear(8, 1)

    @property
    def n_layers(self):
        return self.quantum.n_layers

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)
        x = self.fc1(x)

        x = self.quantum(x)
        x = torch.sigmoid(self.fc2(x))

        return x
