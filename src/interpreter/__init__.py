from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from frontend.compress import create_empty_circuit


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


# ==============================================================================
# Helper
# ==============================================================================


def transpile_two(qc: QuantumCircuit) -> QuantumCircuit:
    """Convert a circuit to only use 2-input gates.

    Args:
        qc (QuantumCircuit): Input quantum circuit.

    Returns:
        QuantumCircuit: Quantum circuit that only uses 2-input gates.
    """
    # Create an empty circuit
    qc_p = create_empty_circuit(qc)

    # Transpile original circuit to use only 2-qubit gates
    aer_sim = AerSimulator()
    basis_gates = ["id", "u1", "u2", "u3", "cx", "cu1", "cu2", "cu3"]
    qc_trans = transpile(qc, backend=aer_sim, basis_gates=basis_gates)

    # Accumulate gates
    for gate in qc_trans:
        qc_p.append(gate)
    return qc_p
