import scipy.sparse as sp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from frontend.compress import *
from frontend.tensor import *
from interpreter import Interpreter
from backend.cpu import CPUBackend


class TensorInterpreter(Interpreter):
    """
    The interpreter built based on tensor network
    from this paper: https://arxiv.org/abs/2309.16475.


    Args:
        eps (float, optional): noise factor, can heavily affect the translation. Defaults to None to disable customized noise.
    """

    def __init__(self, eps: float = None) -> None:
        self.eps = eps

    def run(self, qc: QuantumCircuit, num_shots=1024, all_histories=False) -> dict:
        assert qc.num_qubits == 2, "Currently only support 2-qubit system"

        gates = compress_circuit(qc, Compress.parallel)[0]
        unitaries = [sp.csc_matrix(Operator(x)) for x in gates]

        # TODO: for n-qubit case, calculate the actual depth
        n = qc.num_qubits
        D = len(unitaries)
        if self.eps is None:
            self.eps = (D + 1) ** (-0.51)  # originally poly(D) ** (-0.51)

        program = TensorFrontend(self.eps).unitaries_to_program(unitaries, n, D)

        raw_results = CPUBackend().run(program, num_shots)
        if all_histories:
            return raw_results

        results = {}
        valid_states = ["00", "11"]
        for res, count in raw_results.items():
            for i in range(2, 2 * n * D + 1, 2):
                if res[i - 2 : i] not in valid_states:
                    break
            else:
                results.setdefault(res[-n:], 0)
                results[res[-n:]] += count

        return results
