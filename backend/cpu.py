import scipy.sparse as sp
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from tqdm import tqdm

from backend import *


class CPUBackend(Backend):
    def run(
        self,
        adiabatic_program: ClockAdiabaticProgram | PlanarAdiabaticProgram,
        num_shots=1024,
    ) -> dict:
        H_init, H_final = adiabatic_program.compile()
        duration = adiabatic_program.total_time / adiabatic_program.time_steps
        H_of_s = lambda s: sp.linalg.expm(
            (-1j * duration * ((1 - s) * H_init + s * H_final))
        )

        if isinstance(adiabatic_program, PlanarAdiabaticProgram):
            num_all = 2 * 3 * adiabatic_program.num_comp * adiabatic_program.num_round
        else:
            num_all = adiabatic_program.num_comp + adiabatic_program.num_clock
        qc = QuantumCircuit(num_all)
        q_range = list(range(num_all))
        # Hamiltonian Gate is not used because it is slower
        for j in tqdm(range(adiabatic_program.time_steps + 1)):
            qc.unitary(H_of_s(j / adiabatic_program.time_steps).toarray(), q_range)
        qc.measure_all()
        return AerSimulator().run(qc, shots=num_shots).result().get_counts()
