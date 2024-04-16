import scipy.sparse as sp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from frontend.compress import *
from frontend.tensor import *
from interpreter import Interpreter
from backend.cpu import CPUBackend


class TensorInterpreter(Interpreter):

    def run(self, qc: QuantumCircuit, num_shots=10000) -> dict:
        gates = compress_circuit(qc, Compress.parallel)[0]
        unitaries = [sp.csc_matrix(Operator(x)) for x in gates]

        # TODO: for n-qubit case, calculate the actual depth
        n = qc.num_qubits
        D = len(unitaries)
        program = TensorFrontend().unitaries_to_program(unitaries, n, D)

        results = {}
        valid_states = ["00", "11"]
        for res, count in CPUBackend().run(program, num_shots).items():
            for i in range(2, 2 * n * D + 1, 2):
                if res[i - 2 : i] not in valid_states:
                    break
            else:
                results.setdefault(res[-n:], 0)
                results[res[-n:]] += count

        return results


qc = QuantumCircuit(2)
qc.x([0])
print(sorted(TensorInterpreter().run(qc).items()))

qc = QuantumCircuit(2)
qc.x([1])
print(sorted(TensorInterpreter().run(qc).items()))

qc = QuantumCircuit(2)
qc.x([0, 1])
print(sorted(TensorInterpreter().run(qc).items()))
