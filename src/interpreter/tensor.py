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
        program = TensorFrontend().unitaries_to_program(unitaries, qc.num_qubits, len(unitaries))

        for x,y in CPUBackend().run(program, 1024).items():
            print(x, y)


qc = QuantumCircuit(2)
qc.x(1)

TensorInterpreter().run(qc)
