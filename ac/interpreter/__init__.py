from qiskit import QuantumCircuit


class Interpreter:
    def __init__(self) -> None:
        pass

    def run(self, qc: QuantumCircuit, num_shots=1024) -> dict:
        """Run the circuit.

        Args:
            qc (QuantumCircuit): Quantum circuit to run.
            num_shots (int, optional): Number of shots to run the circuit for. Defaults to 1024.

        Returns:
            dict: Histogram of qubit measurements.
        """
        raise NotImplementedError()
    