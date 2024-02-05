from frontend.clock import ClockFrontend
from test.test_hamiltonian_lang import *


# NOTE: this is pretty much my very 1st raw implementation
class TestClockFrontend(unittest.TestCase):
    def encode_unitary(
        self, unitary: sp.spmatrix, bwd: str, fwd: str, start: int, bound: int
    ):
        assert isinstance(unitary, sp.spmatrix)
        assert len(bwd) == len(fwd)

        return sp.kron(unitary, project_state(fwd, bwd, start, bound)) + sp.kron(
            unitary.conj().transpose(), project_state(bwd, fwd, start, bound)
        )

    def test_H_clock(self):
        clock = sp.csc_matrix((exp2_n, exp2_n))
        for l in range(1, L):
            clock += project_state("01", "01", l, L)
        H_clock = sp.kron(sp.eye(exp2_n), clock)

        assert_sp_matrix_equal(
            H_clock,
            compile_expr(ClockFrontend("5").gen_H_clock(n, L)),
        )

        H_clock = L**6 * sp.kron(sp.eye(exp2_n), clock)

        assert_sp_matrix_equal(
            H_clock,
            compile_expr(ClockFrontend("3").gen_H_clock(n, L)),
        )

    def test_H_input(self):
        compu = sp.csc_matrix((exp2_n, exp2_n))
        for i in range(1, n + 1):
            compu += project_state("1", "1", i, n)
        H_input = sp.kron(compu, project_state("0", "0", 1, L))

        assert_sp_matrix_equal(
            H_input,
            compile_expr(ClockFrontend("5").gen_H_input(n, L)),
        )

    def test_H_clockinit(self):
        H_clockinit = sp.kron(sp.eye(exp2_n), project_state("1", "1", 1, L))

        assert_sp_matrix_equal(
            H_clockinit,
            compile_expr(ClockFrontend("5").gen_H_clockinit(n, L)),
        )

    def test_H_l_sum_part_check_clock(self):
        check_clock = project_state("00", "00", 1, L)
        check_clock += project_state("10", "10", 1, L)
        check_clock += project_state("10", "10", L - 1, L)
        check_clock += project_state("11", "11", L - 1, L)
        for l in range(2, L):
            check_clock += project_state("100", "100", l - 1, L)
            check_clock += project_state("110", "110", l - 1, L)

        check_clock = sp.kron(sp.eye(exp2_n), check_clock)
        print(ClockFrontend("5").gen_H_l_sum_part_check_clock(n, L))
        assert_sp_matrix_equal(
            check_clock,
            compile_expr(ClockFrontend("5").gen_H_l_sum_part_check_clock(n, L)),
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
        encoded_unitaries = sp.csc_matrix((exp2_all, exp2_all))

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
            compile_expr(ClockFrontend("5").gen_H_l_sum_part_unitary(new_L, gates)),
        )

        gates = [None] + gates
        encoded_unitaries = sp.csc_matrix((exp2_all, exp2_all))
        for l in range(1, new_L + 1):
            encoded_unitaries += self.encode_unitary(gates[l], "0", "1", l, new_L)

        gates.pop(0)
        assert_sp_matrix_equal(
            encoded_unitaries,
            compile_expr(ClockFrontend("3").gen_H_l_sum_part_unitary(new_L, gates)),
        )
