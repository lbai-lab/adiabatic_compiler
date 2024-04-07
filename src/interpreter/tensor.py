import scipy.sparse as sp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from frontend.compress import *
from frontend.tensor import *
from interpreter import Interpreter
from backend.cpu import CPUBackend


class TensorInterpreter(Interpreter):

    def run(self, qc: QuantumCircuit, num_shots=1024) -> dict:
        gates = compress_circuit(qc, Compress.parallel)[0]
        unitaries = [sp.csc_matrix(Operator(x)) for x in gates]

        # TODO: for n-qubit case, calculate the actual depth
        n = qc.num_qubits
        program = TensorFrontend().unitaries_to_program(unitaries, n, len(unitaries))

        results = {}
        for res, count in CPUBackend().run(program, 1024).items():
            real_res = res[-n:]
            results.setdefault(real_res, 0)
            results[real_res] += count

        print(results)


qc = QuantumCircuit(2)
qc.x(0)
qc.x(1)

TensorInterpreter().run(qc)
