from qiskit import QuantumCircuit
from itertools import permutations
from qiskit.converters import circuit_to_dag, dag_to_circuit

# TODO: put every compress in single file


class Compress:
    no = "no"
    all = "all"
    all_half = "all-half"
    parallel = "parallel"
    two_qubit = "2-qubit"
    two_qubit_perm = "2-qubit-permute"


def create_empty_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    """Create an empty circuit.

    Args:
        qc (QuantumCircuit): Input quantum circuit.

    Returns:
        QuantumCircuit: An empty quantum circuit with the same shape as the input.
    """
    return QuantumCircuit(qc.qubits, global_phase=qc.global_phase)


# TODO: take care of barriers
def compress_circuit(qc: QuantumCircuit, compress: Compress):
    qubit_map = None  # a mapping used for two-qubit-permute compress

    if compress == Compress.no:
        gates = []
        for gate in qc:
            circ = create_empty_circuit(qc)
            circ.append(gate)
            gates.append(circ)

    elif compress == Compress.all:
        gates = [qc]

    elif compress == Compress.all_half:
        half = qc.size() // 2
        circ_1 = create_empty_circuit(qc)
        circ_2 = create_empty_circuit(qc)
        for gate in qc[:half]:
            circ_1.append(gate)
        for gate in qc[half:]:
            circ_2.append(gate)
        gates = [circ_1, circ_2]

    elif compress == Compress.parallel:
        gates = []
        for layer in [dag_to_circuit(x["graph"]) for x in circuit_to_dag(qc).layers()]:
            gates.append(layer)

    elif compress in (Compress.two_qubit, Compress.two_qubit_perm):
        best_size = float("inf")
        best_gates, best_qubit_map = [], []

        for new_pos in (
            permutations(range(qc.num_qubits))
            if compress == Compress.two_qubit_perm
            else [tuple(range(qc.num_qubits))]
        ):
            pos = list(range(qc.num_qubits))
            index_map = {i: i for i in pos}

            for i, v in enumerate(new_pos):
                if pos[i] != v:
                    i_new = index_map[v]

                    pos[i], pos[i_new] = pos[i_new], pos[i]

                    index_map[v] = i
                    index_map[pos[i_new]] = i_new

            # map qubit based on the above permuation index map
            # to understand this, I suggest to look into qc.reverse_bits()
            qubit_map = {}
            for qb in qc.qubits:
                qubit_map[index_map[qb.index]] = qb
            qc_temp = QuantumCircuit(qc.qubits, global_phase=qc.global_phase)
            for instruction in qc.data:
                qc_temp._append(
                    instruction.replace(
                        qubits=[
                            qubit_map[qc_temp.find_bit(qubit).index]
                            for qubit in instruction.qubits
                        ]
                    )
                )

            cur_gates = []
            idx_set = set()
            circ = create_empty_circuit(qc)

            for gate in qc_temp:
                # index may be deprecated in the future
                indices = [x.index for x in gate.qubits]
                idx_set.update(indices)

                if len(idx_set) > 2:
                    cur_gates.append(circ)
                    if len(cur_gates) >= best_size:
                        print("skip worse case, haven't triggered so far")
                        continue
                    circ = create_empty_circuit(qc)
                    idx_set = set(indices)

                circ.append(gate)

            cur_gates.append(circ)

            if len(cur_gates) < best_size:
                best_size = len(cur_gates)
                best_gates = cur_gates
                best_qubit_map = list(new_pos)

            gates = best_gates
            qubit_map = best_qubit_map

    else:
        raise ValueError(f"Invalid compress option: {compress}")

    return gates, qubit_map
