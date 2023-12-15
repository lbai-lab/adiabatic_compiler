from frontend import *


class Backend:
    """Backend class."""

    def __init__(self):
        pass

    def run(self, adiabatic_program, num_shots) -> dict:
        """Produce an executable for a simulator from a program.

        Args:
            adiabatic_program (AdiabaticProgram): The program.
            num_shots (int): The number of run shots.

        Returns:
            dict: The run results.
        """
        raise NotImplementedError()
