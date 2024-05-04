import scipy.sparse as sp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit.singleton import SingletonGate

from backend.cpu import CPUBackend
from frontend.compress import *
from frontend.planar_clock import PlanarClockFrontend
from interpreter import *


class PlanarClockInterpreter(Interpreter):
    """
    The interpreter built based on Section 5  (2-local)
    from this paper: https://arxiv.org/abs/quant-ph/0405098.
    """

    # ensure the circuit only have 2-qubit neighbor gates
    def _pre_process(self, input_qc: QuantumCircuit):
        qc = input_qc.copy_empty_like()

        # use transplie to keep gates in order
        for ins, qubits, _ in input_qc:
            if len(qubits) > 1:
                if len(qubits) > 2:
                    raise ValueError("not 1- or 2-qubit gates only")
                indices = sorted(q._index for q in qubits)
                i1, i2 = indices[0], indices[1]

                if i1 + 1 != i2:
                    new_i1 = i2 - 1
                    for i in range(i1, new_i1):
                        qc.swap(i, i + 1)
                    qc.append(ins.replace(qubits=[qc.qubits[new_i1], qc.qubits[i2]]))
                    for i in range(new_i1, i1, -1):
                        qc.swap(i, i - 1)

                else:
                    qc.append(ins)
            else:
                qc.append(ins)

        return qc

    def run(self, gate_methods: list[list[SingletonGate]], num_shots=1024) -> dict:
        """Convert and run the circuit.

        Args:
            qc (QuantumCircuit): Quantum circuit to run.
            num_shots (int, optional): Number of shots to run the circuit for. Defaults to 1024.

        Returns:
            dict: Results without interpretation, since this is only partially finished
        """
        assert len(gate_methods) > 0, "No gates is provided"

        num_qubits = len(gate_methods[0])
        assert num_qubits >= 2, "Require at least a 2-qubit system"

        for i, round in enumerate(gate_methods):
            if len(round) != num_qubits:
                raise ValueError(f"Inconsistent number of gates for round #{i}")

        I_mat = sp.csc_matrix(sp.eye(2))
        gates = []
        for round in gate_methods:
            for j, gate in enumerate(round):
                if j == 0:
                    qc = QuantumCircuit(1)
                    qc.append(gate(), [0])
                else:
                    qc = QuantumCircuit(2)
                    qc.append(gate(), [0, 1])

                gates.append(sp.csc_matrix(Operator(qc)))

            for _ in range(num_qubits):
                gates.append(I_mat)

        program = PlanarClockFrontend().unitaries_to_program(gates)

        return CPUBackend().run(program, num_shots)
