"""Eight-qubit variational layer: RX angle encoding, configurable-depth
BasicEntanglerLayers (RX + CNOT ring), eight Pauli-Z expectation values.

The default depth (3) is the original architecture; checkpoints saved before
depth became configurable therefore load unchanged.
"""
import pennylane as qml
import torch.nn as nn

N_QUBITS = 8
DEFAULT_LAYERS = 3
n_qubits = N_QUBITS  # backward-compatible name used by the September code

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    # BasicEntanglerLayers infers the number of layers from weights.shape[0].
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="X")
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS), rotation=qml.RX)
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class QuantumLayer(nn.Module):
    def __init__(self, n_layers=DEFAULT_LAYERS):
        super().__init__()
        if isinstance(n_layers, bool) or int(n_layers) != n_layers or n_layers < 1:
            raise ValueError(f'n_layers must be a positive integer, got {n_layers!r}')
        self.n_layers = int(n_layers)
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, {"weights": (self.n_layers, N_QUBITS)})

    def forward(self, x):
        return self.q_layer(x)
