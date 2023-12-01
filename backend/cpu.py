import numpy as np
import scipy.sparse as sp
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from tqdm import tqdm

from backend import *


class CPUBackend(Backend):
    def run(self, adiabatic_program: AdiabaticProgram, num_shots=1024) -> dict:
        H_init, H_final = adiabatic_program.compile()
        duration = adiabatic_program.total_time / adiabatic_program.time_steps
        H_of_s = lambda s: sp.linalg.expm(
            (-1j * duration * ((1 - s) * H_init + s * H_final))
        )

        # TODO: reimplement fake simulator
        # if use_fake_sim:  # simulate directly on unitary matrix, faster
        #     qc = []
        #     for j in range(steps + 1):
        #         print(f"Constructing time step #{j}", end="\r")
        #         qc.append(H_of_s(j / steps))
        # prob = sp.csc_matrix((1, self.H_init.shape[0]))
        # prob[0, 0] = 1
        # for mat in self.qc_adia:
        #     prob *= mat

        # # get probability (U * U_dagger)
        # prob = sp.csc_matrix.multiply(prob, prob.conjugate()).getrow(0)
        # idx = prob.nonzero()
        # prob = prob[idx].getA1().astype(np.float64)

        # # np.random.choice is not good at handling non-precise probability
        # res = {}
        # fmt = f"{{:0{self.num_all}b}}"
        # for i, v in zip(idx[1], np.random.multinomial(1024, prob / sum(prob))):
        #     if v:
        #         res[fmt.format(i)] = v

        num_all = adiabatic_program.num_comp + adiabatic_program.num_clock
        qc = QuantumCircuit(num_all)
        q_range = list(range(num_all))
        # Hamiltonian Gate is not used because it is slower
        for j in tqdm(range(adiabatic_program.time_steps + 1)):
            qc.unitary(H_of_s(j / adiabatic_program.time_steps).toarray(), q_range)
        qc.measure_all()
        return AerSimulator().run(qc, shots=num_shots).result().get_counts()
