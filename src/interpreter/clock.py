from typing import Literal

import scipy.sparse as sp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from backend.cpu import *
from frontend.clock import *
from interpreter import *


class ClockInterpreter(Interpreter):
    """
    The interpreter built based on Section 3  (5-local) and Section 4 (3-local)
    from this paper: https://arxiv.org/abs/quant-ph/0405098,
    integrated with muiltiple options and optimizations when intialized.


    Args:
        locality (Literal["5", "3"], optional): 5- or 3-local translation. Defaults to "5".
        transpile_to_two (bool, optional): Transpile the input circuit to contain only 2-qubit gates. Defaults to False.
        info (bool, optional): Print all information. Defaults to True.
        end_i (int, optional): Number of identities added to the end of input circuit. Defaults to 0.
        compress (Compress, optional): Option to compress the input circuit. Defaults to Compress.no.
    """

    def __init__(
        self,
        locality: Literal["5", "3"] = "5",
        transpile_to_two=False,
        info=True,
        end_i=0,
        compress: Compress = Compress.no,
    ):
        super().__init__()
        self.locality = locality  # k-local Hamiltonian
        self.transpile_to_two = transpile_to_two  # transpile first?
        self.info = info  # should we print?
        self.end_i = end_i  # end with identity?
        self.compress = compress  # compress options

    def run(self, qc: QuantumCircuit, num_shots=1024, all_histories=False) -> dict:
        """Convert and run the circuit.

        Args:
            qc (QuantumCircuit): Quantum circuit to run.
            num_shots (int, optional): Number of shots to run the circuit for. Defaults to 1024.
            all_histories (bool, optional): Return uninterpreted results with clock states. Defaults to False.

        Returns:
            dict: Results after running the circuit.
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

        n = adiabatic_program.num_data
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
