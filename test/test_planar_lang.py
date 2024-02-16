from test import *

from language.planar_hamiltonian import *
from tqdm import tqdm

# the smallest example: 2 qubits and 1 round
n = 2
R = 1

total_particles = n * (R + 1)
total_qubits = 3 * total_particles

hori_all_qubits = 3 * (n + 1)
size_hori_all = 2**hori_all_qubits
hori_middle_qubits = 3 * (n - 1)

SPAN_ALL = [START, END] + SPAN_FIRST + SPAN_SECOND


class TestPlanarHamiltonianLanguage(unittest.TestCase):
    def test_Identity(self):
        assert_sp_matrix_equal(
            reify3(n, R, Identity(1, 0)), sp.eye(2 ** (3 * n * (R + 1)))
        )

    def test_SingProj(self):
        for state in tqdm(SPAN_ALL, leave=False):
            mat = init_square_matrix(8)
            idx = int(encode3(state), 2)
            mat[idx, idx] = 1

            for i in range(n):
                for r in range(R + 1):
                    start = 3 * (i + r * n)
                    assert_sp_matrix_equal(
                        reify3(n, R, SingProj(state, i + 1, r)),
                        kron_I(mat, start, total_qubits - start - 3),
                    )

    def test_VertProj(self):
        for state1 in tqdm(SPAN_ALL, leave=False):
            for state2 in tqdm(SPAN_ALL, leave=False):
                mat = init_square_matrix(64)
                idx = int(encode3(state1) + encode3(state2), 2)
                mat[idx, idx] = 1

                for i in range(n - 1):  # vertically 1 less
                    for r in range(R + 1):
                        start = 3 * (i + r * n)
                        assert_sp_matrix_equal(
                            reify3(n, R, VertProj(state1, state2, i + 1, r)),
                            kron_I(mat, start, total_qubits - start - 6),
                        )

    def test_HoriProj(self):
        for state1 in tqdm(SPAN_ALL, leave=False):
            for state2 in tqdm(SPAN_ALL, leave=False):

                mat = init_square_matrix(size_hori_all)
                s1, s2 = encode3(state1), encode3(state2)
                for middle in bin_strs(hori_middle_qubits):
                    idx = int(s1 + middle + s2, 2)
                    mat[idx, idx] = 1

                for i in range(n):
                    for r in range(R):  # horizontally 1 less
                        start = 3 * (i + r * n)
                        assert_sp_matrix_equal(
                            reify3(n, R, HoriProj(state1, state2, i + 1, r)),
                            kron_I(mat, start, total_qubits - start - hori_all_qubits),
                        )

    def test_HoriSymProject(self):
        for state11 in tqdm(SPAN_ALL, leave=False):
            s11 = encode3(state11)
            for state12 in tqdm(SPAN_ALL, leave=False):
                s12 = encode3(state12)
                for state21 in tqdm(SPAN_ALL, leave=False):
                    s21 = encode3(state21)
                    for state22 in tqdm(SPAN_ALL, leave=False):
                        s22 = encode3(state22)

                        mat = init_square_matrix(size_hori_all)
                        for middle in bin_strs(hori_middle_qubits):
                            idx1 = int(s11 + middle + s12, 2)
                            idx2 = int(s21 + middle + s22, 2)
                            mat[idx1, idx2] = 1
                            mat[idx2, idx1] = 1

                        for i in range(n):
                            for r in range(R):  # horizontally 1 less
                                start = 3 * (i + r * n)
                                assert_sp_matrix_equal(
                                    reify3(
                                        n,
                                        R,
                                        HoriSymProject(
                                            [state11, state12],
                                            [state21, state22],
                                            i + 1,
                                            r,
                                        ),
                                    ),
                                    kron_I(
                                        mat,
                                        start,
                                        total_qubits - start - hori_all_qubits,
                                    ),
                                )
