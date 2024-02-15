from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from interpreter import Interpreter


class GateBasedInterpreter(Interpreter):
    """
    Gate-based interpreter. Used to check correctness.
    """

    def run(self, qc: QuantumCircuit, num_shots=1024) -> dict:
        """Run the circuit in gate-based platform.

        Args:
            qc (QuantumCircuit): Quantum circuit to run.
            num_shots (int, optional): Number of shots to run the circuit for. Defaults to 1024.

        Returns:
            dict: Results after running the circuit.
        """
        # prepare
        sim = AerSimulator()
        qc = qc.copy()
        qc.measure_all()

        # ensure the circuit is runnable
        qc = transpile(qc, backend=sim)

        # return the results
        return sim.run(qc, shots=num_shots).result().get_counts()
