from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from tqdm import tqdm

from backend import *


class CPUBackend(Backend):
    def run(self, adiabatic_program: AdiabaticProgram, num_shots: int) -> dict:
        H_init, H_final = adiabatic_program.compile()
        duration = adiabatic_program.total_time / adiabatic_program.time_steps
        H_of_s = lambda s: sp.linalg.expm(
            (-1j * duration * ((1 - s) * H_init + s * H_final))
        )

        qc = QuantumCircuit(adiabatic_program.num_all)
        q_range = list(range(qc.num_qubits))
        # Hamiltonian Gate is not used because it is slower
        for j in tqdm(range(adiabatic_program.time_steps + 1)):
            qc.unitary(H_of_s(j / adiabatic_program.time_steps).toarray(), q_range)
        qc.measure_all()
        return AerSimulator().run(qc, shots=num_shots).result().get_counts()
