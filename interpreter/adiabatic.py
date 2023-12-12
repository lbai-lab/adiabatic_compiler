import numpy as np
import scipy.sparse as sp
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

from backend.cpu import *
from frontend.clock import *
from interpreter import Interpreter

from typing import Literal

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


def spmat_to_str(spmat: sp.spmatrix) -> str:
    return (
        "["
        + ", ".join(
            [
                "[" + ", ".join([str(x) for x in row]) + "]"
                for row in np.array(spmat.todense())
            ]
        )
        + "]"
    )


# ==============================================================================
# Interpreter
# ==============================================================================


class AdiabaticInterpreter(Interpreter):
    def __init__(
        self,
        locality: Literal["5", "3"] = "5",
        compress: Compress = Compress.no,
        end_i=0,
        transpile_to_two=False,
        info=True,
    ):
        super().__init__()
        self.locality = locality  # k-local Hamiltonian
        self.compress = compress  # compress options
        self.end_i = end_i  # end with identity?
        self.transpile_to_two = transpile_to_two  # transpile first?
        self.info = info  # should we print?

    def run(self, qc: QuantumCircuit, num_shots=1024, all_histories=False) -> dict:
        """Run the circuit.

        Args:
            qc (QuantumCircuit): Quantum circuit to run.
            num_shots (int, optional): Number of shots to run the circuit for. Defaults to 1024.
            all_histories (bool, optional): Return results of clock states as well. Defaults to False.

        Returns:
            dict: Histogram of qubit measurements.
        """
        qc_orig = qc

        # 1. transpile to get "L 2-qubit gates" as stated in the theorem
        if self.transpile_to_two:
            qc = transpile_two(qc_orig)

        # 2. compress gates if prompted
        gates, qubit_map = compress_circuit(qc, self.compress)

        # 3. an optimization mentioned in the paper
        if self.end_i > 0:
            for _ in range(self.end_i):
                gates.append(create_empty_circuit(qc_orig))
        while len(gates) < 2:
            print("INFO: not enough unitary, adding 1 identity")
            gates.append(create_empty_circuit(qc_orig))
            self.end_i += 1

        if self.info:
            print(
                "\n*** INFO ***\n"
                f"\nlocality mode = {self.locality}"
                f"\ncompress mode = {self.compress}"
                f"\nqubit map = {qubit_map}"
                f"\n# of gates = {qc.size()}"
                f"\n# of computation qubits (n) = {qc_orig.num_qubits}"
                f"\n# of clock qubits (L) = {len(gates)}"
                f"\n# of ending identities = {self.end_i}"
                "\n\n*** END OF INFO ***\n"
            )

        # 4. convert unitaries into the adibatic program
        adiabatic_program = ClockFrontend(self.locality).unitaries_to_program(
            [sp.csc_matrix(Operator(x)) for x in gates]
        )

        n = adiabatic_program.num_comp
        L = adiabatic_program.num_clock

        res = CPUBackend().run(adiabatic_program, num_shots=num_shots)

        # interpret the results
        if qubit_map:
            new_res = {}
            for k, v in res.items():
                if v:
                    key = ""
                    for i in qubit_map:
                        key += k[i]
                    key += k[n:]
                    new_res[key] = v
            res = new_res

        if all_histories:
            return {f"({k[n:]})\n{k[:-L]}": v for k, v in res.items()}

        final_res = {}
        clock_pos = n + L - 1 - self.end_i
        for state in res:
            if state[clock_pos] == "1":  # take advantage of the clock pattern
                comp_state = state[:n]
                final_res.setdefault(comp_state, 0)
                final_res[comp_state] += res[state]

        return final_res
