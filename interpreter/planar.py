from qiskit import QuantumCircuit, transpile
from interpreter import Interpreter
from frontend.planar_clock import PlanarClockFrontend
import scipy.sparse as sp
from frontend.compress import *
from backend.cpu import CPUBackend

# from backend.reify import reify_2local, reify3
# from backend.cpu import AdiabaticCPUExecutable


class PlanarClockInterpreter(Interpreter):
    def __init__(self) -> None:
        """Gate-based interpreter."""
        super().__init__()

    # def _get_indices(self, qc: QuantumCircuit) -> list[int]:
    #     return sorted(q.index for q in qc.qubits)

    # # ensure the circuit only have 2-qubit neighbor gates
    # def _pre_process(self, input_qc: QuantumCircuit):
    #     qc = input_qc.copy_empty_like()

    #     # use transplie to keep gates in order
    #     for gate in input_qc:
    #         if len(gate.qubits) > 1:
    #             if len(gate.qubits) > 2:
    #                 raise ValueError("not 1- or 2-qubit gates only")
    #             indices = self._get_indices(gate)
    #             i1, i2 = indices[0], indices[1]

    #             if i1 + 1 != i2:
    #                 new_i1 = i2 - 1
    #                 for i in range(i1, new_i1):
    #                     qc.swap(i, i + 1)
    #                 qc.append(gate.replace(qubits=[qc.qubits[new_i1], qc.qubits[i2]]))
    #                 for i in range(new_i1, i1, -1):
    #                     qc.swap(i, i - 1)

    #             else:
    #                 qc.append(gate)
    #         else:
    #             qc.append(gate)

    #     return list(transpile(qc))

    # TODO
    def run(self, num_shots=1024) -> dict:
        raise NotImplementedError()
