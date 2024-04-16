import math

import scipy.sparse as sp
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

from frontend import *


class TensorAdiabaticProgram(AdiabaticProgram):
    def __init__(
        self,
        H_init: sp.csc_matrix,
        H_final: sp.csc_matrix,
        total_time: float,
        time_steps: int,
        num_all: int,
    ) -> None:
        super().__init__(H_init, H_final, total_time, time_steps, num_all)

    def compile(self) -> tuple[sp.csc_matrix, sp.csc_matrix]:
        return self.H_init, self.H_final


class TensorFrontend(Frontend):
    __phi_I = (sp.csc_matrix([[1], [0], [0], [1]])) / math.sqrt(2)
    __phi_X = sp.kron(sp.eye(2), sp.csc_matrix([[0, 1], [1, 0]])) @ __phi_I
    __phi_Z = sp.kron(sp.eye(2), sp.csc_matrix([[1, 0], [0, -1]])) @ __phi_I
    __phi_XZ = (
        sp.kron(
            sp.eye(2),
            sp.csc_matrix([[0, 1], [1, 0]]) @ sp.csc_matrix([[1, 0], [0, -1]]),
        )
        @ __phi_I
    )

    def __init__(self, eps: float = 2):
        self.lamb = sp.csc_matrix(
            eps * self.project(self.__phi_I)
            + (
                self.project(self.__phi_X)
                + self.project(self.__phi_Z)
                + self.project(self.__phi_XZ)
            )
        )

    def lamb_n(self, n: int) -> sp.csc_matrix:
        res = self.lamb
        for _ in range(n - 1):
            res = sp.kron(res, self.lamb)
        return sp.csc_matrix(res)

    def project(self, arr) -> sp.csc_matrix:
        arr = sp.csc_matrix(arr).reshape((-1, 1))
        return arr @ arr.T

    def gen_H_in(self, n: int, D: int) -> sp.csc_matrix:
        in_columns = 2 * n
        lamb = self.lamb_n(n)

        states = []
        for i in range(2**in_columns):
            count = 0
            b = bin(i)[2:].zfill(in_columns)
            for j in range(0, in_columns, 2):
                if b[j] == "1":
                    count += 1

            states.append(count)

        return sp.csc_matrix(
            sp.kron(lamb @ self.project(states) @ lamb, sp.eye(2 ** (n * (2 * D - 1))))
        )

    def gen_H_out(self, n: int, D: int) -> sp.csc_matrix:
        states = [bin(i)[2:].count("1") for i in range(2**n)]

        return sp.csc_matrix(sp.kron(sp.eye(2 ** (n * 2 * D)), self.project(states)))

    def encode_U(self, U: sp.spmatrix, is_boundary: bool) -> sp.csc_matrix:
        qc = QuantumCircuit(4)
        qc.append(UnitaryGate(U.toarray()), [1, 3])
        encoded_U = sp.eye(16) - self.project(
            Operator(qc).data @ sp.kron(self.__phi_I, self.__phi_I)
        )

        if is_boundary:
            qc = QuantumCircuit(6)
            qc.append(UnitaryGate(encoded_U.toarray(), check_input=False), [1, 3, 4, 5])
        else:
            qc = QuantumCircuit(8)
            qc.append(UnitaryGate(encoded_U.toarray(), check_input=False), [1, 3, 4, 6])

        return sp.csc_matrix(Operator(qc))

    def gen_H_prop(self, n: int, D: int, Us: list[sp.spmatrix]) -> sp.csc_matrix:
        columns = 2 * D + 1
        total = n * columns

        # last unitary
        lamb_end = sp.kron(self.lamb_n(2), sp.eye(4))
        qc = QuantumCircuit(total)
        qc.append(
            UnitaryGate(
                (lamb_end @ self.encode_U(Us[-1], True) @ lamb_end).toarray(),
                check_input=False,
            ),
            range(total - 6, total),
        )

        h_u_sum = sp.csc_matrix(Operator(qc))

        # non-last unitary
        lamb_4 = self.lamb_n(4)
        for i in range(D - 1):
            q_i = 2 * i

            qc = QuantumCircuit(total)
            qc.append(
                UnitaryGate(
                    (lamb_4 @ self.encode_U(Us[i], False) @ lamb_4).toarray(),
                    check_input=False,
                ),
                range(q_i, q_i + 8),
            )

            h_u_sum += sp.csc_matrix(Operator(qc))

        return h_u_sum

    def unitaries_to_program(
        self, Us: list[sp.spmatrix], num_qubits: int, depth: int
    ) -> TensorAdiabaticProgram:
        if len(Us) < 1:
            raise ValueError(f"Require at least one unitary matrix")
        assert num_qubits == 2, "Currently only support 2-qubits circuit"

        H_in = self.gen_H_in(num_qubits, depth)
        H_out = self.gen_H_out(num_qubits, depth)
        H_prop = self.gen_H_prop(num_qubits, depth, Us)

        total = num_qubits * (2 * depth + 1)
        return TensorAdiabaticProgram(H_in, H_prop + H_out, 2**total, total, total)
