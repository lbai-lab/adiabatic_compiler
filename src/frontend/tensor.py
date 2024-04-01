import itertools

import numpy as np
import scipy.sparse as sp

from frontend import *


class TensorAdiabaticProgram(AdiabaticProgram):
    def __init__(
        self,
        H_init: sp.spmatrix,
        H_final: sp.spmatrix,
        total_time: float,
        time_steps: int,
        num_all: int,
    ) -> None:
        super().__init__(H_init, H_final, total_time, time_steps, num_all)

    def compile(self) -> tuple[sp.spmatrix, sp.spmatrix]:
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

    def lamb_n(self, n: int) -> sp.spmatrix:
        res = self.lamb
        for _ in range(n - 1):
            res = sp.kron(res, self.lamb)
        return res

    def project_bell(self, n: int, i1: int, i2: int):
        result = [
            int(arr[i1] == arr[i2]) for arr in itertools.product([0, 1], repeat=n)
        ]
        return sp.diags(result)

    def gen_H_in(self, n: int, D: int) -> sp.spmatrix:
        lamb = self.lamb_n(n)
        size_one_column = 2**n
        penalize_one = [bin(i)[2:].count("1") for i in range(size_one_column)]

        return sp.kron(
            lamb.dot(sp.kron(sp.diags(penalize_one), sp.eye(size_one_column))).dot(
                lamb
            ),
            sp.eye(2 ** ((2 * D - 1) * n)),
        )

    def gen_H_out(self, n: int, D: int) -> sp.spmatrix:
        penalize_one = [bin(i)[2:].count("1") for i in range(2**n)]

        return sp.kron(
            sp.eye(2 ** (2 * D * n)),
            sp.diags(penalize_one),
        )

    def gen_H_prop(self, n: int, D: int, Us: list[sp.spmatrix]) -> sp.spmatrix:
        # doing element-wise multiply achieven the "kron" in 1d matrix
        # arr = set(int(x, 2) for x in ["0000", "0101", "1010", "1111"])
        # bell = sp.diags([1 if i in arr else 0 for i in range(16)])
        bell = self.project_bell(4, 0, 2).dot(self.project_bell(4, 1, 3)) / 2

        h_us = []
        # TODO: for more Us
        lamb_4 = self.lamb_n(4)
        phi_U_project = sp.kron(sp.eye(4), Us[0]).dot(bell)
        h_u = lamb_4.dot(
            sp.eye(256) - sp.kron(sp.eye(4), sp.kron(phi_U_project, sp.eye(4)))
        ).dot(lamb_4)
        h_us.append(sp.kron(h_u, sp.eye(4)))

        lamb_end = sp.kron(self.lamb_n(2), sp.eye(4))
        h_end = lamb_end.dot(sp.eye(64) - sp.kron(sp.eye(4), sp.eye(16).dot(bell))).dot(
            lamb_end
        )
        h_us.append(sp.kron(sp.eye(16), h_end))

        for h in h_us:
            print(h.shape)

        return sum(h_us)

    def unitaries_to_program(
        self, Us: list[sp.spmatrix], num_qubits: int, depth: int
    ) -> TensorAdiabaticProgram:
        if len(Us) < 1:
            raise ValueError(f"Require at least one unitary matrix")
        assert num_qubits == 2, "Currently only support 2-qubits circuit"

        depth += 1

        H_in = self.gen_H_in(num_qubits, depth)
        H_out = self.gen_H_out(num_qubits, depth)
        H_prop = self.gen_H_prop(num_qubits, depth, Us)

        total = num_qubits * (2 * depth + 1)
        return TensorAdiabaticProgram(H_prop, H_prop, 2**total, total, total)
