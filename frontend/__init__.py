from hamiltonian_lang import *


class Frontend:
    def __init__(self) -> None:
        pass

    def unitaries_to_program(self, Us: list[sp.spmatrix]) -> HamExpr:
        """Translate a list of unitaries into an adiabatic program.

        Args:
            Us (list[sp.spmatrix]): list of unitaries.

        Returns:
            AdiabaticProgram: The adiabatic program.
        """
        pass
