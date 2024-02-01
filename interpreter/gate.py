from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from interpreter import Interpreter


class GateBasedInterpreter(Interpreter):
    def __init__(self) -> None:
        """Gate-based interpreter."""
        super().__init__()

    def run(self, qc: QuantumCircuit, num_shots=1024) -> dict:
        """Run the circuit.

        Args:
            qc (QuantumCircuit): Quantum circuit to run.
            num_shots (int, optional): Number of shots to run the circuit for. Defaults to 1024.

        Returns:
            dict: Histogram of qubit measurements.
        """
        # prepare
        sim = AerSimulator()
        qc = qc.copy()
        qc.measure_all()

        # ensure the circuit is runnable
        qc = transpile(qc, backend=sim)

        # return the results
        return sim.run(qc, shots=num_shots).result().get_counts()
