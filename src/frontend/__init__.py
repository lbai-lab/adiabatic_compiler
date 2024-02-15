from language import *


class AdiabaticProgram:
    """
    The adiabatic program stores the required data for AQC.
    It should be returned from Frontend after converted,
    and then passed to the Backend to run as adiabatic.

    Args:
        H_init (ExpressionBase): The language expression of the initial Hamiltonian.
        H_final (ExpressionBase): The language expression of the final Hamiltonian.
        total_time (float): The running time for the AQC.
        time_steps (int): The number of time steps.
        num_all (int): The number of all qubits, including both state and clock qubits.
    """

    def __init__(
        self,
        H_init: ExpressionBase,
        H_final: ExpressionBase,
        total_time: float,
        time_steps: int,
        num_all: int,
    ) -> None:
        self.H_init = H_init
        self.H_final = H_final
        self.total_time = total_time
        self.time_steps = time_steps
        self.num_all = num_all

    def compile(self) -> tuple[sp.spmatrix, sp.spmatrix]:
        """
        Compile the language expression of the Hamiltonians.

        Returns:
            tuple[sp.spmatrix, sp.spmatrix]: The compiled Hamiltonians.
        """
        raise NotImplementedError()


class Frontend:
    """
    Frontend is where the real translation happens.
    It is the implementation of the proposed translation of the papers we found.
    """

    def unitaries_to_program(self, Us: list[sp.spmatrix]) -> AdiabaticProgram:
        """
        Translate a list of unitaries into an adiabatic program.

        Args:
            Us (list[sp.spmatrix]): list of unitaries.

        Returns:
            AdiabaticProgram: The adiabatic program.
        """
        pass
