"""
Phase-VI: Quantum Circuit Specification (PhaseVI 2.8, 2.17).
Reconstructs the deployed QMLP circuit and the re-uploading VQC comparator circuit
exactly as defined in the code, then reports circuit resources: qubit count,
trainable parameters, encoding, entangler, logical depth, total gate count,
two-qubit (CNOT) gate count, and transpiled depth/CX to a standard basis.
Backend is exact Statevector simulation (no shots, no noise) -> deterministic.
"""
import os, sys, json
from pathlib import Path
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P))
from vqc_model import ReuploadingVQC


def build_qmlp(n=8, L=4):
    """Deployed QMLP: H+Ry encode once, then L layers of Rz - brickwall CNOT - Ry."""
    qc = QuantumCircuit(n)
    rng = np.random.RandomState(0)
    x = rng.uniform(0.1, np.pi - 0.1, n)          # nonzero feature angles (avoid identity removal)
    th = rng.uniform(0.1, np.pi - 0.1, 2 * n * L)  # nonzero trainable angles
    for i in range(n):
        qc.h(i); qc.ry(float(x[i]), i)
    for layer in range(L):
        for i in range(n):
            qc.rz(float(th[layer * n + i]), i)
        if layer % 2 == 0:
            for i in range(0, n - 1, 2):
                qc.cx(i, i + 1)
        else:
            for i in range(1, n - 1, 2):
                qc.cx(i, i + 1)
            qc.cx(n - 1, 0)
        for i in range(n):
            qc.ry(float(th[L * n + layer * n + i]), i)
    return qc


def spec(qc, n_trainable, encoding, name):
    ops = dict(qc.count_ops())
    cx = ops.get("cx", 0)
    depth = qc.depth()
    size = sum(ops.values())
    # transpile to a common basis for a device-agnostic compiled depth
    tq = transpile(qc, basis_gates=["u3", "cx"], optimization_level=1)
    tops = dict(tq.count_ops())
    return {
        "name": name, "n_qubits": qc.num_qubits, "n_trainable_params": n_trainable,
        "encoding": encoding, "entangler": "brick-wall CNOT (alternating, with wrap)",
        "logical_depth": int(depth), "total_gates": int(size), "cx_gates": int(cx),
        "single_qubit_rotations": int(sum(v for k, v in ops.items() if k in ("ry", "rz", "rx", "h", "u3", "u"))),
        "op_counts": {k: int(v) for k, v in ops.items()},
        "transpiled_basis": "[u3, cx]", "transpiled_depth": int(tq.depth()),
        "transpiled_cx": int(tops.get("cx", 0)),
    }


qmlp = build_qmlp(8, 4)
vqc = ReuploadingVQC(n_qubits=8, n_layers=4)._qc

out = {
    "qiskit_version": qiskit.__version__,
    "backend": "qiskit.quantum_info.Statevector (exact statevector, no shots, no noise model)",
    "simulator_determinism": "exact statevector -> deterministic; no simulator seed / shot count needed",
    "qmlp_deployed": spec(qmlp, 64, "H + Ry(x) encode-once (data uploaded once)", "Deployed QuantumAI QMLP (circuit-only readout)"),
    "vqc_comparator": spec(vqc, 64, "Ry(x)+Rz(x) re-uploaded every layer", "Re-uploading VQC comparator"),
    "notes": {
        "qmlp_classical_head": "9 classical params (8 weights + bias); deployed decision reads Fisher/PCA features linearly, circuit readout is diagnostic only",
        "readout": "per-qubit P(|1>) marginals from Statevector (QMLP); single qubit-0 marginal (VQC)",
    },
}
json.dump(out, open(P / "results_phaseVI" / "circuit_spec.json", "w"), indent=1)
for k in ("qmlp_deployed", "vqc_comparator"):
    s = out[k]
    print(f"{s['name']}: qubits={s['n_qubits']} params={s['n_trainable_params']} "
          f"depth={s['logical_depth']} gates={s['total_gates']} cx={s['cx_gates']} "
          f"transpiled_depth={s['transpiled_depth']} ({s['transpiled_basis']})")
print("qiskit", out["qiskit_version"])
print("op_counts qmlp:", out["qmlp_deployed"]["op_counts"])
print("op_counts vqc :", out["vqc_comparator"]["op_counts"])
