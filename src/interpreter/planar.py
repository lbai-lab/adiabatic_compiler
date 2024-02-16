import scipy.sparse as sp
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

from backend.cpu import CPUBackend
from frontend.compress import *
from frontend.planar_clock import PlanarClockFrontend
from interpreter import Interpreter


class PlanarClockInterpreter(Interpreter):
    """
    The interpreter built based on Section 5  (2-local)
    from this paper: https://arxiv.org/abs/quant-ph/0405098.
    """

    def run(self, num_shots=1024) -> dict:
        """Convert and run the circuit.

        Args:
            qc (QuantumCircuit): Quantum circuit to run.
            num_shots (int, optional): Number of shots to run the circuit for. Defaults to 1024.
            all_histories (bool, optional): Return unprocessed results with clock states. Defaults to False.

        Returns:
            dict: Results after running the circuit.
        """

        # currently use an example circuit of the bell state
        gates = []
        qc = QuantumCircuit(1)
        qc.x(0)
        gates.append(Operator(qc))
        qc = QuantumCircuit(2)
        # qc.cx(0, 1)
        qc.x(1)
        gates.append(Operator(qc))
        gates.append(Operator(QuantumCircuit(1)))
        gates.append(Operator(QuantumCircuit(1)))

        program = PlanarClockFrontend().unitaries_to_program(
            [sp.csc_matrix(x) for x in gates]
        )

        return CPUBackend().run(program, num_shots)
