import scipy.sparse as sp
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

from backend.cpu import CPUBackend
from frontend.compress import *
from frontend.planar_clock import PlanarClockFrontend
from interpreter import Interpreter


class PlanarClockInterpreter(Interpreter):
    def __init__(self) -> None:
        """Gate-based interpreter."""
        super().__init__()

    def run(self, num_shots=1024) -> dict:
        # currently use an example circuit of the bell state
        gates = []
        qc = QuantumCircuit(1)
        qc.h(0)
        gates.append(Operator(qc))
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        gates.append(Operator(qc))
        gates.append(Operator(QuantumCircuit(1)))
        gates.append(Operator(QuantumCircuit(1)))

        program = PlanarClockFrontend().unitaries_to_program(
            [sp.csc_matrix(x) for x in gates]
        )

        # return CPUBackend().run(program)
