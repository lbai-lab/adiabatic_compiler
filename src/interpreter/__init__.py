from qiskit import QuantumCircuit


class Interpreter:
    """ 
    The parent class of all interpreters.
    It is the entry point of the translation.
    """

    def __init__(self) -> None:
        pass

    def run(self, qc: QuantumCircuit, num_shots=1024) -> dict:
        """Convert (if defined) and run the circuit.

        Args:
            qc (QuantumCircuit): Quantum circuit to run.
            num_shots (int, optional): Number of shots to run the circuit for. Defaults to 1024.

        Returns:
            dict: Results after running the circuit.
        """
        raise NotImplementedError()
