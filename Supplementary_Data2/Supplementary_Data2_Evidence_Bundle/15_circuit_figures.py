"""
Export  circuit diagrams from the ACTUAL Qiskit circuit
objects used in the study (not schematics):

  - qcnn_pool_circuit.pdf : the validation-selected QCNN-style pooling circuit
    (main text, Section 'Improving the circuit-driven path')
  - reup_L1_circuit.pdf, reup_L4_circuit.pdf : re-uploading VQC family
    (Supplementary Materials)
  - qmlp_deployed_circuit.pdf : the deployed QuantumAI QMLP circuit
    (encode-once, brick-wall) (Supplementary Materials)
"""
import os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent
REPO = Path(os.environ["HYBRID_IDS_REPO"]).resolve()
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(P))

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qarch import make_arch

OUT = P.parent / "figures_phaseVI"
OUT.mkdir(exist_ok=True)


def build_qmlp(n=8, L=4):
    x = ParameterVector("x", n)
    th = ParameterVector(chr(952), 2 * n * L)  # theta
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i); qc.ry(x[i], i)
    for layer in range(L):
        for i in range(n):
            qc.rz(th[layer * n + i], i)
        if layer % 2 == 0:
            for i in range(0, n - 1, 2): qc.cx(i, i + 1)
        else:
            for i in range(1, n - 1, 2): qc.cx(i, i + 1)
            qc.cx(n - 1, 0)
        for i in range(n):
            qc.ry(th[L * n + layer * n + i], i)
    return qc


def draw(qc, fname, fold=28, scale=0.55):
    fig = qc.draw(output="mpl", fold=fold, scale=scale, style={"name": "bw"})
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / fname)


draw(make_arch("qcnn-pool")._qc, "qcnn_pool_circuit.pdf", fold=26, scale=0.6)
draw(make_arch("reup-L1")._qc, "reup_L1_circuit.pdf", fold=26, scale=0.6)
draw(make_arch("reup-L4")._qc, "reup_L4_circuit.pdf", fold=34, scale=0.45)
draw(build_qmlp(), "qmlp_deployed_circuit.pdf", fold=34, scale=0.45)
print("all circuit figures done")
