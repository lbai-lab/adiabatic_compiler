from test import *

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from planar_hamiltonian_lang import *

all_states = [START, END, FIRST0, FIRST1, SECOND0, SECOND1]

gates = []
qc = QuantumCircuit(1)
qc.h(0)
gates.append(Operator(qc))
qc = QuantumCircuit(2)
qc.cx(0, 1)
gates.append(Operator(qc))
gates.append(Operator(QuantumCircuit(1)))
gates.append(Operator(QuantumCircuit(1)))
Us = [sp.csc_matrix(x) for x in gates]

L = len(Us)
n = int(np.log2(Us[1].shape[0]))
R = L // (2 * n)
print(f"n={n}, L={L}, R={R}")


class TestPlanarLanguage(unittest.TestCase):
    # unnecssary since this is a constant
    def test_Identity(self):
        pass

    def test_SingProj(self):
        pass

    def test_VertProj(self):
        pass

    def test_HoriProj(self):
        pass

    def test_HoriSymProject(self):
        pass

    def test_SymUnitary(self):
        pass

    def test_ScalarSum(self):
        pass
