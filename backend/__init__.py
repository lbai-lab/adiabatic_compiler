from frontend import *


class Backend:
    """
    Backend is where the circuit get excuted/running.
    TODO: seprate gate-based and aqc backend...
    """

    def __init__(self):
        pass

    def run(self, adiabatic_program: AdiabaticProgram, num_shots: int) -> dict:
        """
        Run/Execute the adiabatic program.

        Args:
            adiabatic_program (AdiabaticProgram): The program holding all necessary information to run adiabatic quantum computation.
            num_shots (int): The number of run shots.

        Returns:
            dict: Results after running the circuit.
        """
        raise NotImplementedError()
