import scipy.sparse as sp
from itertools import product

# ==============================================================================
# 6-state Particles
# ==============================================================================

START = "START"  # empty
FIRST0 = "FIRST0"  # arrow up
FIRST1 = "FIRST1"  # arrow down
SECOND0 = "SECOND0"  # two arrows up
SECOND1 = "SECOND1"  # two arrows down
END = "END"  # cross

SPAN_FIRST = [FIRST0, FIRST1]
SPAN_SECOND = [SECOND0, SECOND1]


# ==============================================================================
# Planar Hamiltonian Expressions
# ==============================================================================


class PlanarHamExpr:
    """Planar Hamiltonian Expression.

    Example:

    R = 2, C = 2 has a grid layout of R x (C + 1)
        (r=0, c=0)  (r=0, c=1)  (r=0, c=2)
        (r=1, c=0)  (r=1, c=1)  (r=1, c=2)
        (r=2, c=0)  (r=2, c=1)  (r=2, c=2)
    """

    def __init__(self, row=None, col=None) -> None:
        assert isinstance(row, int) or row is None
        assert isinstance(col, int) or col is None
        self.row = row
        self.col = col
        self.scalar = 1


class Identity(PlanarHamExpr):
    """Identity Operator.

    I
    """

    def __init__(self, n: int, row: int, col: int) -> None:
        super().__init__(row=row, col=col)
        self.n = n

    def __str__(self):
        return f"I({self.n})_({self.row}, {self.col})"


class SingProj(PlanarHamExpr):
    """Single 2-local Hamiltonian.

    |p><p|
    """

    def __init__(self, state: str, row: int, col: int) -> None:
        super().__init__(row, col)
        self.state = state

    def __str__(self) -> str:
        return f"P({self.state})_{self.row, self.col}"


class VertProj(PlanarHamExpr):
    """Vertical 2-local Hamiltonian.

    |p_1 > < p_1|
    |p_2 > < p_2|
    """

    def __init__(self, state1: str, state2: str, row: int, col: int) -> None:
        super().__init__(row, col)
        self.state1 = state1
        self.state2 = state2

    def __str__(self) -> str:
        return f"VP({self.state1}, {self.state2})_({self.row}, {self.col})"


class HoriProj(PlanarHamExpr):
    """Horizontal 2-local Hamiltonian.

    |p_1 p_2> <p_1 p_2|
    """

    def __init__(self, state1: str, state2: str, row: int, col: int) -> None:
        super().__init__(row, col)
        self.state1 = state1
        self.state2 = state2

    def __str__(self) -> str:
        return f"HP({self.state1}, {self.state2})_({self.row}, {self.col})"


class HoriSymProject(PlanarHamExpr):
    """Horizontal 2-local Hamiltonian.

    |p_1 p_2><p_3 p_4| + |p_3 p_4><p_1 p_2|
    """

    def __init__(
        self, states1: tuple[str, str], states2: tuple[str, str], row: int, col: int
    ) -> None:
        super().__init__(row, col)
        self.states1 = states1
        self.states2 = states2

    def __str__(self):
        return f"SP({self.states1}, {self.states2})_({self.row}, {self.col})"


class SymUnitary(PlanarHamExpr):
    """Vertical 2-local Hamiltonian that symmetrizes unitary.

    0           U
    U^\dagger   0
    """

    def __init__(self, U: sp.spmatrix, row: int, col: int) -> None:
        assert (U.shape[0] == 2 and row == 1) or (
            U.shape[0] == 4 and row > 1
        ), f"Wrong unitary location, recieved U shape = {U.shape}, row = {row}"
        super().__init__(row, col)
        self.U = U

    def __str__(self) -> str:
        return f"U_{self.row, self.col}"


class ScalarSum(PlanarHamExpr):
    """Sum of 2-local Hamiltonians."""

    def __init__(self, Hs: list[PlanarHamExpr], scalar=1) -> None:
        super().__init__()
        self.Hs = Hs
        self.scalar = scalar

    def __str__(self) -> str:
        text = "\n".join([str(H) for H in self.Hs])

        if self.scalar != 1:
            text = str(self.scalar) + " * " + text

        return text


# ==============================================================================
# Grid
# ==============================================================================


class Grid:
    def __init__(self, rows: int, cols: int, H: PlanarHamExpr) -> None:
        self.rows = rows
        self.cols = cols
        self.H = H
        self.grid: dict[tuple[int, int], list[PlanarHamExpr]] = {}
        self._gridify()

    # NOTE:
    # the rows were in 1-index form following the paper
    # but the columns are not, because col_0 is used as input
    def _gridify(self):
        for r in range(self.rows):
            for c in range(0, self.cols + 1):
                self.grid[r, c] = []

        def go(H: PlanarHamExpr, scalar=1):
            if isinstance(H, PlanarHamExpr):
                H.scalar *= scalar
                if isinstance(H, ScalarSum):
                    for H2 in H.Hs:
                        go(H2, H.scalar)
                else:
                    self.grid[H.row - 1, H.col].append(H)
            else:
                raise ValueError(f"Unexpected type {type(H)}")

        go(self.H, 1)


"""

    q_2        q_8
    |\          |\
    | \q_0      | q_6
    |/          |/
    q_1        q_7

q_5        q_11
|\          |\
| \q_3      | q_9
|/          |/
q_4        q_10

|q_12 q_11 q_10 q_9 q_8 q_7 q_6 q_5 q_4 q_3 q_2 q_1 q_0>
"""


# ==============================================================================
# Utility
# ==============================================================================


def encode3(st: str) -> str:
    """Particle to 3 qubits.

    q_2
    |\
    | \q_0
    |/ 
    q_1
    
    Args:
        st (str): Particle State

    Returns:
        str: Encoding
    """
    if st is START:
        return "000"
    elif st is END:
        return "001"
    elif st is FIRST0:
        return "010"
    elif st is FIRST1:
        return "011"
    elif st is SECOND0:
        return "100"
    elif st is SECOND1:
        return "101"
    else:
        raise ValueError(f"{st} not supported ...")


_F0 = encode3(FIRST0)
_F1 = encode3(FIRST1)
_S0 = encode3(SECOND0)
_S1 = encode3(SECOND1)


def get_particle(n: int, i: int, j: int):
    q_0 = 3 * i + j * 3 * n
    return q_0, q_0 + 1, q_0 + 2


def bin_strs(n: int) -> list[str]:
    return ["".join(p) for p in product("01", repeat=n)]


def kron_I(mat: sp.spmatrix, front: int, back: int):
    assert front >= 0 and back >= 0, "Unexpected negative identity numbers"
    return sp.kron(sp.eye(2**front), sp.kron(mat, sp.eye(2**back)))


def init_square_matrix(size: int) -> sp.csc_matrix:
    return sp.csc_matrix((size, size)) * 1j


# the entry of compile
# different from compile_expr in regular clock
def reify3(n: int, R: int, H: PlanarHamExpr):
    num_particles = n * (R + 1)
    num_qubits = 3 * num_particles
    size = 2**num_qubits
    my_H = init_square_matrix(size)

    for pos, Hs in Grid(n, R, H).grid.items():
        if len(Hs) == 0:
            continue

        k, r = pos
        qs_k_r = get_particle(n, k, r)
        start_idx = qs_k_r[0]
        hori_all_qubits = 3 * (n + 1)
        size_hori_all = 2**hori_all_qubits
        hori_middle_qubits = 3 * (n - 1)

        for H in Hs:
            # TODO: check what Identity.n can do so far
            # but probably just leave it alone
            if isinstance(H, Identity):
                my_H += H.scalar * sp.eye(size)

            elif isinstance(H, SingProj):
                X = init_square_matrix(8)
                idx = int(encode3(H.state), 2)
                X[idx, idx] = 1.0

                my_H += H.scalar * kron_I(X, start_idx, num_qubits - start_idx - 3)

            elif isinstance(H, VertProj):
                X = init_square_matrix(64)
                idx = int(encode3(H.state1) + encode3(H.state2), 2)
                X[idx, idx] = 1.0

                my_H += H.scalar * kron_I(X, start_idx, num_qubits - start_idx - 6)

            elif isinstance(H, HoriProj):
                X = init_square_matrix(size_hori_all)

                s1, s2 = encode3(H.state1), encode3(H.state2)
                for middle in bin_strs(hori_middle_qubits):
                    idx = int(s1 + middle + s2, 2)
                    X[idx, idx] = 1.0

                my_H += H.scalar * kron_I(
                    X, start_idx, num_qubits - start_idx - hori_all_qubits
                )

            elif isinstance(H, HoriSymProject):
                X = init_square_matrix(size_hori_all)
                s1_1, s1_2 = encode3(H.states1[0]), encode3(H.states1[1])
                s2_1, s2_2 = encode3(H.states2[0]), encode3(H.states2[1])

                for middle in bin_strs(hori_middle_qubits):
                    idx1 = int(s1_1 + middle + s1_2, 2)
                    idx2 = int(s2_1 + middle + s2_2, 2)
                    X[idx1, idx2] = 1.0
                    X[idx2, idx1] = 1.0

                my_H += H.scalar * kron_I(
                    X, start_idx, num_qubits - start_idx - hori_all_qubits
                )

            elif isinstance(H, SymUnitary):
                U = -H.U
                if U.shape[0] == 2:
                    X = init_square_matrix(8)
                    X[int(_F0, 2), int(_S0, 2)] = U[0, 0]
                    X[int(_F0, 2), int(_S1, 2)] = U[0, 1]
                    X[int(_F1, 2), int(_S0, 2)] = U[1, 0]
                    X[int(_F1, 2), int(_S1, 2)] = U[1, 1]
                    X += X.conj().transpose()
                    my_H += H.scalar * kron_I(X, start_idx, num_qubits - start_idx - 3)
                elif U.shape[0] == 4:
                    X = init_square_matrix(64)
                    # |bot top>
                    # 00 -> [00, 01, 10, 11]
                    X[int(_F0 + _F0, 2), int(_S0 + _S0, 2)] = U[0, 0]
                    X[int(_F0 + _F0, 2), int(_S0 + _S1, 2)] = U[0, 1]
                    X[int(_F0 + _F0, 2), int(_S1 + _S0, 2)] = U[0, 2]
                    X[int(_F0 + _F0, 2), int(_S1 + _S1, 2)] = U[0, 3]
                    # 01 -> [00, 01, 10, 11]
                    X[int(_F0 + _F1, 2), int(_S0 + _S0, 2)] = U[1, 0]
                    X[int(_F0 + _F1, 2), int(_S0 + _S1, 2)] = U[1, 1]
                    X[int(_F0 + _F1, 2), int(_S1 + _S0, 2)] = U[1, 2]
                    X[int(_F0 + _F1, 2), int(_S1 + _S1, 2)] = U[1, 3]
                    # 10 -> [00, 01, 10, 11]
                    X[int(_F1 + _F0, 2), int(_S0 + _S0, 2)] = U[2, 0]
                    X[int(_F1 + _F0, 2), int(_S0 + _S1, 2)] = U[2, 1]
                    X[int(_F1 + _F0, 2), int(_S1 + _S0, 2)] = U[2, 2]
                    X[int(_F1 + _F0, 2), int(_S1 + _S1, 2)] = U[2, 3]
                    # 11 -> [00, 01, 10, 11]
                    X[int(_F1 + _F1, 2), int(_S0 + _S0, 2)] = U[3, 0]
                    X[int(_F1 + _F1, 2), int(_S0 + _S1, 2)] = U[3, 1]
                    X[int(_F1 + _F1, 2), int(_S1 + _S0, 2)] = U[3, 2]
                    X[int(_F1 + _F1, 2), int(_S1 + _S1, 2)] = U[3, 3]
                    X += X.conj().transpose()
                    my_H += H.scalar * kron_I(X, start_idx, num_qubits - start_idx - 6)

                else:
                    raise ValueError("Unexpected shape (should be 2x2 or 4x4)", U.shape)
            else:
                raise ValueError(f"{type(H)} not implemented")

    return my_H
