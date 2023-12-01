from test import *

from frontend.clock import ClockFrontend

# NOTE: this is pretty much my very 1st raw implementation


class TestClockFrontend(unittest.TestCase):
    def project_state(self, ket: str, bra: str, start: int, bound: int) -> sp.spmatrix:
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

    def encode_unitary(
        self, unitary: sp.spmatrix, bwd: str, fwd: str, start: int, bound: int
    ):
        assert isinstance(unitary, sp.spmatrix)
        assert len(bwd) == len(fwd)

        return sp.kron(unitary, self.project_state(fwd, bwd, start, bound)) + sp.kron(
            unitary.conj().transpose(), self.project_state(bwd, fwd, start, bound)
        )

    def test_H_clock(self):
        clock = sp.lil_matrix((exp2_n, exp2_n))
        for l in range(1, L):
            clock += self.project_state("01", "01", l, L)
        H_clock = sp.kron(sp.eye(exp2_n), clock)

        assert_sp_matrix_equal(
            H_clock,
            compile_expr(ClockFrontend("5")._gen_H_clock(n, L)),
        )

        H_clock = L**6 * sp.kron(sp.eye(exp2_n), clock)

        assert_sp_matrix_equal(
            H_clock,
            compile_expr(ClockFrontend("3")._gen_H_clock(n, L)),
        )

    def test_H_input(self):
        compu = sp.lil_matrix((exp2_n, exp2_n))
        for i in range(1, n + 1):
            compu += self.project_state("1", "1", i, n)
        H_input = sp.kron(compu, self.project_state("0", "0", 1, L))

        assert_sp_matrix_equal(
            H_input,
            compile_expr(ClockFrontend("5")._gen_H_input(n, L)),
        )

    def test_H_clockinit(self):
        H_clockinit = sp.kron(sp.eye(exp2_n), self.project_state("1", "1", 1, L))

        assert_sp_matrix_equal(
            H_clockinit,
            compile_expr(ClockFrontend("5")._gen_H_clockinit(n, L)),
        )

    def test_H_l_sum_part_check_clock(self):
        check_clock = self.project_state("00", "00", 1, L)
        check_clock += self.project_state("10", "10", 1, L)
        check_clock += self.project_state("10", "10", L - 1, L)
        check_clock += self.project_state("11", "11", L - 1, L)
        for l in range(2, L):
            check_clock += self.project_state("100", "100", l - 1, L)
            check_clock += self.project_state("110", "110", l - 1, L)

        check_clock = sp.kron(sp.eye(exp2_n), check_clock)
        print(ClockFrontend("5")._gen_H_l_sum_part_check_clock(n, L))
        assert_sp_matrix_equal(
            check_clock,
            compile_expr(ClockFrontend("5")._gen_H_l_sum_part_check_clock(n, L)),
        )

    def test_H_l_sum_part_unitary(self):
        gates = [
            sp.csc_matrix(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ]
            ),
            sp.csc_matrix(
                [
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ),
            sp.csc_matrix(
                [
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ]
            ),
        ]
        new_n, new_L = 2, 3
        exp2_all = 2 ** (new_n + new_L)
        encoded_unitaries = sp.lil_matrix((exp2_all, exp2_all))

        gates = [None] + gates
        encoded_unitaries += self.encode_unitary(gates[1], "00", "10", 1, new_L)
        encoded_unitaries += self.encode_unitary(
            gates[new_L], "10", "11", new_L - 1, new_L
        )
        for l in range(2, new_L):
            encoded_unitaries += self.encode_unitary(
                gates[l], "100", "110", l - 1, new_L
            )

        gates.pop(0)
        assert_sp_matrix_equal(
            encoded_unitaries,
            compile_expr(ClockFrontend("5")._gen_H_l_sum_part_unitary(new_L, gates)),
        )

        gates = [None] + gates
        encoded_unitaries = sp.lil_matrix((exp2_all, exp2_all))
        for l in range(1, new_L + 1):
            encoded_unitaries += self.encode_unitary(gates[l], "0", "1", l, new_L)

        gates.pop(0)
        assert_sp_matrix_equal(
            encoded_unitaries,
            compile_expr(ClockFrontend("3")._gen_H_l_sum_part_unitary(new_L, gates)),
        )
