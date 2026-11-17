import pennylane as qml
import torch
import torch.nn as nn

# ==========================================
# Quantum Device
# ==========================================

n_qubits = 4

dev = qml.device("default.qubit", wires=n_qubits)

# ==========================================
# Quantum Circuit
# ==========================================

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):

    qml.AngleEmbedding(inputs, wires=range(n_qubits))

    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# ==========================================
# Quantum Layer
# ==========================================

weight_shapes = {"weights": (4, n_qubits, 3)}

quantum_layer = qml.qnn.TorchLayer(
    quantum_circuit,
    weight_shapes
)


# ==========================================
# Hybrid Quantum Neural Network
# ==========================================

class HybridQNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(11, 32)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(32, 4)

        self.quantum = quantum_layer

        self.fc3 = nn.Linear(4, 16)
        self.relu2 = nn.ReLU()

        self.fc4 = nn.Linear(16, 2)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)

        x = self.quantum(x)

        x = self.fc3(x)
        x = self.relu2(x)

        x = self.fc4(x)

        return x


# ==========================================
# Test Model
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("HYBRID QUANTUM NEURAL NETWORK")
    print("=" * 60)

    model = HybridQNN()

    print("\nModel Created Successfully!\n")

    sample = torch.rand(5, 11)

    output = model(sample)

    print("Input Shape :", sample.shape)
    print("Output Shape:", output.shape)

    print("\nHybrid QNN Ready!")