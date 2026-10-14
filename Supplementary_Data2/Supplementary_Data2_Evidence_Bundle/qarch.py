"""
Quantum architecture candidates for the improved-readout / architecture study.

All candidates:
  - consume the q=8 Fisher/PCA features as rotation angles (scaling handled by caller)
  - are trained with the SAME Adam-SPSA budget as the deployed QuantumAI agent
    (300 iterations, batch 5, perturbation 0.25, lr 0.01) for fair comparison
  - expose  marginals_batch(X, weights) -> (n, 8) per-qubit P(|1>) marginals
    and     forward_batch(X, weights)   -> (n,)  single-readout probability
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


class _Base:
    n_qubits = 8

    def __init__(self):
        n2 = 2 ** self.n_qubits
        self._bit_masks = np.array(
            [[(b >> i) & 1 for b in range(n2)] for i in range(self.n_qubits)],
            dtype=np.float64)
        self._readout_mask = self._bit_masks[self.readout_qubit]
        self.weights = None

    def reset_weights(self, seed):
        rng = np.random.RandomState(seed)
        self.weights = rng.uniform(-0.2, 0.2, self.n_params).astype(np.float64)

    def _bind(self, x_vec, weights):
        params = {}
        for xi, xp in zip(x_vec, self._x_params):
            params[xp] = float(xi)
        for wi, tp in zip(weights, self._theta_params):
            params[tp] = float(wi)
        return self._qc.assign_parameters(params)

    def _probs(self, x_vec, weights):
        return Statevector(self._bind(x_vec, weights)).probabilities()

    def marginals_one(self, x_vec, weights=None):
        w = self.weights if weights is None else weights
        return self._bit_masks @ self._probs(x_vec, w)

    def marginals_batch(self, X, weights=None):
        w = self.weights if weights is None else weights
        return np.array([self._bit_masks @ self._probs(x, w) for x in X])

    def _forward_one(self, x_vec, weights):
        return float(self._readout_mask @ self._probs(x_vec, weights))

    def forward_batch(self, X, weights=None):
        w = self.weights if weights is None else weights
        return np.array([self._forward_one(x, w) for x in X], dtype=np.float64)

    def train(self, X, y, seed, max_iter=300, batch_size=5, lr=0.01,
              beta1=0.9, beta2=0.999, perturbation=0.25,
              patience=40, min_delta=5e-4):
        """Adam-SPSA on BCE of the single-qubit readout (same budget as deployed agent)."""
        self.reset_weights(seed)
        rng = np.random.RandomState(seed + 10_000)
        m = np.zeros(self.n_params); v = np.zeros(self.n_params); t = 0
        yf = y.astype(np.float64)
        best_loss = float("inf"); best_w = self.weights.copy(); pc = patience
        for step in range(1, max_iter + 1):
            idx = rng.randint(0, len(X), size=min(batch_size, len(X)))
            Xb, yb = X[idx], yf[idx]
            delta = (2 * rng.randint(0, 2, self.n_params) - 1).astype(np.float64)
            wp = self.weights + perturbation * delta
            wm = self.weights - perturbation * delta
            pp = self.forward_batch(Xb, wp); pm = self.forward_batch(Xb, wm)
            eps = 1e-8
            lp = float(-np.mean(yb * np.log(pp + eps) + (1 - yb) * np.log(1 - pp + eps)))
            lm = float(-np.mean(yb * np.log(pm + eps) + (1 - yb) * np.log(1 - pm + eps)))
            loss = 0.5 * (lp + lm)
            g = (lp - lm) / (2.0 * perturbation * delta)
            t += 1
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            self.weights -= lr * (m / (1 - beta1 ** t)) / (np.sqrt(v / (1 - beta2 ** t)) + 1e-8)
            if loss < best_loss - min_delta:
                best_loss = loss; best_w = self.weights.copy(); pc = patience
            else:
                pc -= 1
                if pc <= 0:
                    break
        self.weights = best_w
        return best_loss


class ReupVQC(_Base):
    """Re-uploading VQC, parameterised depth L (L=4 is the paper's comparator)."""
    readout_qubit = 0

    def __init__(self, n_layers=4):
        self.n_layers = n_layers
        self.n_params = 2 * self.n_qubits * n_layers
        self.name = f"reup-L{n_layers}"
        n, L = self.n_qubits, n_layers
        x = ParameterVector("x", n)
        th = ParameterVector("th", self.n_params)
        qc = QuantumCircuit(n)
        for layer in range(L):
            for i in range(n):
                qc.ry(x[i], i); qc.rz(x[i], i)
            if layer % 2 == 0:
                for i in range(0, n - 1, 2): qc.cx(i, i + 1)
            else:
                for i in range(1, n - 1, 2): qc.cx(i, i + 1)
                qc.cx(n - 1, 0)
            for i in range(n):
                qc.ry(th[layer * n + i], i)
                qc.rz(th[L * n + layer * n + i], i)
        self._qc, self._x_params, self._theta_params = qc, list(x), list(th)
        super().__init__()


class QCNNPool(_Base):
    """QCNN-style circuit: conv two-qubit blocks + CNOT pooling 8->4->2->1.
    Data encoded once (Ry+Rz); readout from the final surviving qubit.
    7 conv blocks x 6 params = 42 trainable parameters (< the 64 of reup-L4).
    """
    readout_qubit = 7

    def __init__(self):
        self.n_layers = 3
        self.name = "qcnn-pool"
        n = self.n_qubits
        x = ParameterVector("x", n)
        stages = [[(0, 1), (2, 3), (4, 5), (6, 7)], [(1, 3), (5, 7)], [(3, 7)]]
        n_blocks = sum(len(s) for s in stages)
        self.n_params = 6 * n_blocks
        th = ParameterVector("th", self.n_params)
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.ry(x[i], i); qc.rz(x[i], i)
        k = 0
        for stage in stages:
            for (a, b) in stage:
                qc.ry(th[k], a); qc.ry(th[k + 1], b)
                qc.cx(a, b)
                qc.ry(th[k + 2], a); qc.ry(th[k + 3], b)
                qc.cx(b, a)
                qc.rz(th[k + 4], a); qc.rz(th[k + 5], b)
                k += 6
            for (a, b) in stage:  # pooling: fold control qubit into partner
                qc.cx(a, b)
        self._qc, self._x_params, self._theta_params = qc, list(x), list(th)
        super().__init__()


def make_arch(key):
    if key == "qcnn-pool":
        return QCNNPool()
    if key.startswith("reup-L"):
        return ReupVQC(int(key.split("L")[1]))
    raise ValueError(key)
