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
    """
    Implement clock translation from Section 2 from this paper:
    https://arxiv.org/abs/2309.16475.

    Args:
        eps (float): The noise of this translation.
    """

    def __init__(self, eps: float):
        phi_I = (sp.csc_matrix([[1], [0], [0], [1]])) / math.sqrt(2)
        phi_X = sp.kron(sp.eye(2), sp.csc_matrix([[0, 1], [1, 0]])) @ phi_I
        phi_Z = sp.kron(sp.eye(2), sp.csc_matrix([[1, 0], [0, -1]])) @ phi_I
        phi_XZ = (
            sp.kron(
                sp.eye(2),
                sp.csc_matrix([[0, 1], [1, 0]]) @ sp.csc_matrix([[1, 0], [0, -1]]),
            )
            @ phi_I
        )

        self.lamb = sp.csc_matrix(
            eps * self.project(phi_I)
            + (self.project(phi_X) + self.project(phi_Z) + self.project(phi_XZ))
        )

    def lamb_n(self, n: int) -> sp.csc_matrix:
        """
        Kronecker product of the injective map, denoted as Λ (lambda).

        Args:
            n (int): Number of qubit system of input circuit.
        """

        res = self.lamb
        for _ in range(n - 1):
            res = sp.kron(res, self.lamb)
        return sp.csc_matrix(res)

    def project(self, arr) -> sp.csc_matrix:
        """
        The projector, is computed as |x><x|.

        Args:
            arr (list): 1d-array representation of the quantum state.
        """
        arr = sp.csc_matrix(arr).reshape((-1, 1))
        return arr @ arr.T

    def gen_H_in(self, n: int, D: int) -> sp.csc_matrix:
        """
        H_in, at the bottom of page 6, as the first term of H_parent.
        Penalize non-input state.

        Args:
            n (int): Number of qubit system of input circuit.
            D (int): Depth of the input circuit.
        """

        in_columns = 2 * n
        state_1 = sp.diags([0, 1])

        states = sp.diags([0] * (2**in_columns))
        for i in range(0, in_columns, 2):
            states += sp.kron(
                sp.eye(2**i), sp.kron(state_1, sp.eye(2 ** (in_columns - i - 1)))
            )

        lamb = self.lamb_n(n)
        return sp.csc_matrix(
            sp.kron(lamb @ states @ lamb, sp.eye(2 ** (n * (2 * D - 1))))
        )

    def gen_H_out(self, n: int, D: int) -> sp.csc_matrix:
        """
        H_out, at the bottom of page 6.
        Used to check the output column.

        Args:
            n (int): Number of qubit system of input circuit.
            D (int): Depth of the input circuit.
        """

        state_1 = sp.diags([0, 1])

        states = sp.diags([0] * (2**n))
        for i in range(n):
            states += sp.kron(sp.eye(2**i), sp.kron(state_1, sp.eye(2 ** (n - i - 1))))

        return sp.csc_matrix(sp.kron(sp.eye(2 ** (n * 2 * D)), states))

    def encode_U(self, U: sp.spmatrix, is_boundary: bool) -> sp.csc_matrix:
        """
        Encoding unitary with the gate teleportation.

        Args:
            U (sp.spmatrix): The unitary.
            is_boundary (bool): Speical case for boudary, which is the last unitary.
        """

        arr = [0] * 16
        for x in ["0000", "1111", "0101", "1010"]:
            arr[int(x, 2)] = 0.5

        projected_U = self.project(
            sp.kron(sp.eye(4), U) @ sp.csc_matrix(arr).reshape(-1, 1)
        )

        if is_boundary:
            qc = QuantumCircuit(6)
            qc.append(
                UnitaryGate(projected_U.toarray(), check_input=False), [1, 3, 4, 5]
            )
        else:
            qc = QuantumCircuit(8)
            qc.append(
                UnitaryGate(projected_U.toarray(), check_input=False), [1, 3, 4, 6]
            )

        return sp.eye(2**qc.num_qubits) - sp.csc_matrix(Operator(qc))

    def gen_H_prop(self, n: int, D: int, Us: list[sp.spmatrix]) -> sp.csc_matrix:
        """
        H_prop, at the bottom of page 6, as the second term of H_parent.
        Encode the unitary into the injective map within the tensor network.

        Args:
            n (int): Number of qubit system of input circuit.
            D (int): Depth of the input circuit.
            Us (list[sp.spmatrix]): List of unitaries.
        """

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
            range(total - 6, total),  # TODO this is for 2-qubit circuit only
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
                range(q_i, q_i + 8), # TODO this is for 2-qubit circuit only
            )

            h_u_sum += sp.csc_matrix(Operator(qc))

        return h_u_sum

    def unitaries_to_program(
        self, Us: list[sp.spmatrix], num_qubits: int, depth: int
    ) -> TensorAdiabaticProgram:
        if len(Us) < 1:
            raise ValueError(f"Require at least one unitary matrix")

        H_in = self.gen_H_in(num_qubits, depth)
        H_out = self.gen_H_out(num_qubits, depth)
        H_prop = self.gen_H_prop(num_qubits, depth, Us)

        total = num_qubits * (2 * depth + 1)
        return TensorAdiabaticProgram(
            H_in, H_in + H_prop + H_out, 2**total, total, total
        )
