import numpy as np
import scipy.sparse as sp

from frontend import *


class TensorAdiabaticProgram(AdiabaticProgram):
    pass


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

    def lamb_n(self, n: int):
        res = self.lamb
        for _ in range(n - 1):
            res = sp.kron(res, self.lamb)
        return res

    def gen_H_in_and_out(self, n: int, D: int):
        size_one_column = 2**n
        penalize_one = [bin(i)[2:].count("1") for i in range(size_one_column)]
        lamb = self.lamb_n(n)

        return (
            sp.kron(
                lamb.dot(sp.kron(sp.diags(penalize_one), sp.eye(size_one_column))).dot(
                    lamb
                ),
                sp.eye(2 ** ((2 * D - 1) * n)),
            ),
            sp.kron(sp.eye(2 ** (2 * D * n)), sp.diags(penalize_one)),
        )

    def gen_H_prop(self, n: int, D: int, Us: list[sp.spmatrix]):
        # TODO: last layer is implied as identity gates, read paper
        return

    def unitaries_to_program(
        self, Us: list[sp.spmatrix], n: int, D: int
    ) -> TensorAdiabaticProgram:
        if len(Us) < 1:
            raise ValueError(f"Require at least one unitary matrix")
        assert n == 2, "Currently only support 2-qubits circuit"

        H_in, H_out = self.gen_H_in_and_out(n, D)
        H_prop = self.gen_H_prop(n, D, Us)

        print(H_in.shape, H_out.shape)
