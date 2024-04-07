import numpy as np
import scipy.sparse as sp

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

    def __init__(self, eps: float = 0.1):
        phi_I = ((np.array([[1], [0], [0], [1]])) / np.sqrt(2)).astype(np.complex128)
        phi_X = np.dot(np.kron(np.eye(2), np.array([[0, 1], [1, 0]])), phi_I)
        phi_Z = np.dot(np.kron(np.eye(2), np.array([[1, 0], [0, -1]])), phi_I)
        phi_XZ = np.dot(
            np.kron(
                np.eye(2),
                np.dot(np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, -1]])),
            ),
            phi_I,
        )

        self.lamb = sp.csc_matrix(
            eps * np.outer(phi_I, phi_I)
            + (
                np.outer(phi_X, phi_X)
                + np.outer(phi_Z, phi_Z)
                + np.outer(phi_XZ, phi_XZ)
            )
        )

        bell_state = set(int(x, 2) for x in ["0000", "0101", "1010", "1111"])
        self.bell = sp.csc_matrix([[0.5 if i in bell_state else 0] for i in range(16)])

    def lamb_n(self, n: int) -> sp.csc_matrix:
        res = self.lamb
        for _ in range(n - 1):
            res = sp.kron(res, self.lamb)
        return sp.csc_matrix(res)

    def project_phi_U(self, U: sp.csc_matrix) -> sp.csc_matrix:
        phi_U = sp.kron(sp.eye(4), U).dot(self.bell)
        return sp.csc_matrix(phi_U.dot(phi_U.T))

    def gen_H_in(self, n: int, D: int) -> sp.csc_matrix:
        lamb = self.lamb_n(n)
        size_one_column = 2**n
        penalize_one = [bin(i)[2:].count("1") for i in range(size_one_column)]

        return sp.csc_matrix(
            sp.kron(
                lamb.dot(sp.kron(sp.diags(penalize_one), sp.eye(size_one_column))).dot(
                    lamb
                ),
                sp.eye(2 ** ((2 * D - 1) * n)),
            )
        )

    def gen_H_out(self, n: int, D: int) -> sp.csc_matrix:
        penalize_one = [bin(i)[2:].count("1") for i in range(2**n)]

        return sp.csc_matrix(
            sp.kron(
                sp.eye(2 ** (2 * D * n)),
                sp.diags(penalize_one),
            )
        )

    def gen_H_prop(self, n: int, D: int, Us: list[sp.csc_matrix]) -> sp.csc_matrix:
        total = n * (2 * D + 1)

        lamb_end = sp.kron(self.lamb_n(2), sp.eye(4))
        h_u = lamb_end.dot(
            sp.eye(64) - sp.kron(sp.eye(4), self.project_phi_U(Us[-1]))
        ).dot(lamb_end)
        h_u_sum = sp.kron(sp.eye(2 ** (total - 6)), h_u)

        lamb_4 = self.lamb_n(4)
        for i, U in enumerate(Us[:-1]):
            h_u = lamb_4.dot(
                sp.eye(256)
                - sp.kron(sp.eye(4), sp.kron(self.project_phi_U(U), sp.eye(4)))
            ).dot(lamb_4)
            h_u_sum += sp.kron(
                sp.eye(2 ** (4 * i)), sp.kron(h_u, sp.eye(2 ** (total - 4 * i - 8)))
            )

        return sp.csc_matrix(h_u_sum)

    def unitaries_to_program(
        self, Us: list[sp.csc_matrix], num_qubits: int, depth: int
    ) -> TensorAdiabaticProgram:
        if len(Us) < 1:
            raise ValueError(f"Require at least one unitary matrix")
        assert num_qubits == 2, "Currently only support 2-qubits circuit"

        H_in = self.gen_H_in(num_qubits, depth)
        H_out = self.gen_H_out(num_qubits, depth)
        H_prop = self.gen_H_prop(num_qubits, depth, Us)

        total = num_qubits * (2 * depth + 1)
        return TensorAdiabaticProgram(
            H_in, H_in + H_out + H_prop, 2**total, total, total
        )
