import scipy.sparse as sp

from hamiltonian_lang import HamExpr, compile_expr
from planar_hamiltonian_lang import PlanarHamExpr, reify3


class ClockAdiabaticProgram:
    def __init__(
        self,
        num_state: int,
        num_clock: int,
        H_init: HamExpr,
        H_final: HamExpr,
        total_time: float,
        time_steps: int,
    ):
        self.num_state = num_state
        self.num_clock = num_clock
        self.H_init = H_init
        self.H_final = H_final
        self.total_time = total_time
        self.time_steps = time_steps

    def compile(self):
        return compile_expr(self.H_init), compile_expr(self.H_final)


class PlanarAdiabaticProgram:
    def __init__(
        self,
        num_state: int,
        num_round: int,
        num_clock: int,
        H_init: PlanarHamExpr,
        H_final: PlanarHamExpr,
        total_time: float,
        time_steps: int,
    ):
        self.num_state = num_state
        self.num_round = num_round
        self.num_clock = num_clock
        self.H_init = H_init
        self.H_final = H_final
        self.total_time = total_time
        self.time_steps = time_steps

    def compile(self):
        return (
            reify3(self.num_state, self.num_round, self.H_init),
            reify3(self.num_state, self.num_round, self.H_final),
        )


class Frontend:
    def __init__(self) -> None:
        pass

    def unitaries_to_program(self, Us: list[sp.spmatrix]):
        """Translate a list of unitaries into an adiabatic program.

        Args:
            Us (list[sp.spmatrix]): list of unitaries.

        Returns:
            AdiabaticProgram: The adiabatic program.
        """
        pass
