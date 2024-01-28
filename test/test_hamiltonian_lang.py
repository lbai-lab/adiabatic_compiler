from test import *

from ac.hamiltonian_lang import *

n = 4
exp2_n = 2**n

L = 4
exp2_L = 2**n


def project_state(ket: str, bra: str, start: int, bound: int) -> sp.spmatrix:
    assert len(ket) == len(bra), "ket/bra has different length"
    assert start >= 1, "start index is less than 1, it should be 1-indexed"
    end = start - 1 + len(ket)
    assert end <= bound, "'len(ket/bra) + start' exceeds the ending bound"

    size = 2 ** len(ket)
    ket_bra = sp.lil_matrix((size, size))
    ket_bra[int(ket, 2), int(bra, 2)] = 1

    return sp.kron(
        sp.kron(
            sp.eye(2 ** (start - 1)),
            ket_bra,
        ),
        sp.eye(2 ** (bound - end)),
    )


class TestHamiltonianLanguage(unittest.TestCase):
    # unnecssary since this is a constant
    def test_Identity(self):
        for i in range(1, n + 1):
            assert_sp_matrix_equal(compile_expr(Identity(i)), sp.eye(2**i))

    # unnecssary since this is a constant
    def test_ProjectState(self):
        for i in range(exp2_n):
            state = bin(i)[2:].zfill(n)
            zeros = sp.lil_matrix((exp2_n, exp2_n))
            zeros[i, i] = 1

            assert_sp_matrix_equal(compile_expr(ProjectState(state)), zeros)

    def test_KronDiagonal(self):
        expr = Identity(1)
        for _ in range(n - 1):
            expr = KronDiagonal(expr, Identity(1))

        assert_sp_matrix_equal(compile_expr(expr), sp.eye(exp2_n))

    def test_Summation(self):
        expr = Summation([ProjectState(bin(i)[2:].zfill(n)) for i in range(exp2_n)])

        assert_sp_matrix_equal(compile_expr(expr), sp.eye(exp2_n))

    def test_ScalarMultiply(self):
        for i in range(1, 10):
            i /= 5
            assert_sp_matrix_equal(
                compile_expr(ScalarMultiply(-1, Identity(n))),
                -1 * sp.eye(exp2_n),
            )

    def test_EncodeUnitary(self):
        bwd_str, fwd_str = "00", "10"
        U = sp.csc_matrix(  # CNOT gates
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ]
        )

        EncodeUnitary(U, 2, L, bwd_str, fwd_str)
        assert_sp_matrix_equal(
            compile_expr(EncodeUnitary(U, 2, L, bwd_str, fwd_str)),
            sp.kron(U, project_state(fwd_str, bwd_str, 2, L))
            + sp.kron(U.conj().transpose(), project_state(bwd_str, fwd_str, 2, L)),
        )
