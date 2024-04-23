from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from tqdm import tqdm
from qiskit.circuit.library import UnitaryGate

from backend import *


class CPUBackend(Backend):
    def run(self, adiabatic_program: AdiabaticProgram, num_shots: int) -> dict:
        H_init, H_final = adiabatic_program.compile()
        duration = adiabatic_program.total_time / adiabatic_program.time_steps
        H_of_s = lambda s: sp.linalg.expm(
            (-1j * duration * ((1 - s) * H_init + s * H_final))
        )

        q_range = list(range(adiabatic_program.num_all))

        try:
            qc = QuantumCircuit(adiabatic_program.num_all)
            for j in tqdm(range(adiabatic_program.time_steps + 1)):
                qc.append(
                    UnitaryGate(H_of_s(j / adiabatic_program.time_steps).toarray()),
                    q_range,
                )
        except ValueError:
            print("\n*** WARNING ***")
            print("unitary check failed, now force it to run without checking")
            print("This can be due to index offset when encoding unitary")
            print("*** END OF WARNING ***\n")
            qc = QuantumCircuit(adiabatic_program.num_all)
            for j in tqdm(range(adiabatic_program.time_steps + 1)):
                qc.append(
                    UnitaryGate(
                        H_of_s(j / adiabatic_program.time_steps).toarray(),
                        check_input=False,
                    ),
                    q_range,
                )

        qc.measure_all()
        return AerSimulator().run(qc, shots=num_shots).result().get_counts()
