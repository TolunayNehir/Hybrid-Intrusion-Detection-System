"""
Re-uploading Variational Quantum Classifier (VQC) comparator.

Unlike the deployed QMLP (agents/quantum_ai/quantum_module.py), which encodes the
input ONCE (H + Ry(x)) and then applies pure parameterised unitary layers, this VQC
RE-UPLOADS the data at every layer (Ry(x_i); Rz(x_i) repeated each of L layers,
consistent with the "VQC v4" style referenced in the repo's arch_config.json:
"Ry+Rz(x) -> CX -> Ry+Rz(theta) x L"). The classification decision is read directly
from the circuit's own qubit-0 marginal probability -- there is  NO
classical-linear head, unlike the deployed QuantumAI path.

Trained with the same Adam-SPSA optimizer already used elsewhere in the repo, at the
same resource budget (300 iterations, batch 5, perturbation 0.25, lr 0.01) for a fair,
comparably-resourced comparison.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


class ReuploadingVQC:
    def __init__(self, n_qubits: int = 8, n_layers: int = 4, readout_qubit: int = 0):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.readout_qubit = readout_qubit
        self.n_params = 2 * n_qubits * n_layers  # Ry+Rz trainable per qubit per layer

        self._qc, self._x_params, self._theta_params = self._build_circuit()

        n2 = 2 ** n_qubits
        # Marginal-extraction mask for the single readout qubit only.
        self._readout_mask = np.array(
            [(b >> readout_qubit) & 1 for b in range(n2)], dtype=np.float64
        )

        self.weights = None  # set via reset_weights(seed)

    def reset_weights(self, seed: int):
        rng = np.random.RandomState(seed)
        self.weights = rng.uniform(-0.2, 0.2, self.n_params).astype(np.float64)

    def _build_circuit(self):
        n, L = self.n_qubits, self.n_layers
        x = ParameterVector("x", n)
        th = ParameterVector("th", 2 * n * L)  # [Ry block (n*L) | Rz block (n*L)] per layer, layer-major

        qc = QuantumCircuit(n)
        for layer in range(L):
            # ---- re-upload data every layer ----
            for i in range(n):
                qc.ry(x[i], i)
                qc.rz(x[i], i)
            # ---- brick-wall entangler (same topology as the deployed QMLP) ----
            if layer % 2 == 0:
                for i in range(0, n - 1, 2):
                    qc.cx(i, i + 1)
            else:
                for i in range(1, n - 1, 2):
                    qc.cx(i, i + 1)
                qc.cx(n - 1, 0)
            # ---- trainable rotations ----
            for i in range(n):
                qc.ry(th[layer * n + i], i)
                qc.rz(th[L * n + layer * n + i], i)
        return qc, list(x), list(th)

    def _forward_one(self, x_vec: np.ndarray, weights: np.ndarray) -> float:
        params = {}
        for xi, xp in zip(x_vec, self._x_params):
            params[xp] = float(xi)
        for wi, tp in zip(weights, self._theta_params):
            params[tp] = float(wi)
        sv = Statevector(self._qc.assign_parameters(params))
        probs = sv.probabilities()
        return float(self._readout_mask @ probs)  # P(readout qubit = 1)

    def forward_batch(self, X: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
        if weights is None:
            weights = self.weights
        return np.array([self._forward_one(x, weights) for x in X], dtype=np.float64)

    def train(self, X: np.ndarray, y: np.ndarray, seed: int,
              max_iter: int = 300, batch_size: int = 5,
              lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999,
              perturbation: float = 0.25, patience: int = 40, min_delta: float = 5e-4):
        self.reset_weights(seed)
        rng = np.random.RandomState(seed + 10_000)  # independent stream for SPSA/minibatch

        adam_m = np.zeros(self.n_params)
        adam_v = np.zeros(self.n_params)
        adam_t = 0

        y_float = y.astype(np.float64)
        best_loss = float("inf")
        best_weights = self.weights.copy()
        patience_cnt = patience

        for step in range(1, max_iter + 1):
            n = min(batch_size, len(X))
            idx = rng.randint(0, len(X), size=n)
            X_batch, y_batch = X[idx], y_float[idx]

            delta = (2 * rng.randint(0, 2, self.n_params) - 1).astype(np.float64)
            w_plus = self.weights + perturbation * delta
            w_minus = self.weights - perturbation * delta

            p_plus = self.forward_batch(X_batch, w_plus)
            p_minus = self.forward_batch(X_batch, w_minus)
            eps = 1e-8
            loss_p = float(-np.mean(y_batch * np.log(p_plus + eps) + (1 - y_batch) * np.log(1 - p_plus + eps)))
            loss_m = float(-np.mean(y_batch * np.log(p_minus + eps) + (1 - y_batch) * np.log(1 - p_minus + eps)))
            loss = (loss_p + loss_m) * 0.5

            grads = (loss_p - loss_m) / (2.0 * perturbation * delta)
            adam_t += 1
            adam_m = beta1 * adam_m + (1 - beta1) * grads
            adam_v = beta2 * adam_v + (1 - beta2) * grads ** 2
            m_hat = adam_m / (1 - beta1 ** adam_t)
            v_hat = adam_v / (1 - beta2 ** adam_t)
            self.weights -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

            if loss < best_loss - min_delta:
                best_loss = loss
                best_weights = self.weights.copy()
                patience_cnt = patience
            else:
                patience_cnt -= 1
                if patience_cnt <= 0:
                    break

        self.weights = best_weights
        return best_loss
