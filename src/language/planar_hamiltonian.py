from language import *

from itertools import product

# ==============================================================================
# 6-state Particles
# ==============================================================================

START = "START"  # empty
SINGLE_UP = "SINGLE_UP"  # arrow up
SINGLE_DOWN = "SINGLE_DOWN"  # arrow down
DOUBLE_UP = "DOUBLE_UP"  # two arrows up
DOUBLE_DOWN = "DOUBLE_DOWN"  # two arrows down
END = "END"  # cross

SPAN_FIRST = [SINGLE_UP, SINGLE_DOWN]
SPAN_SECOND = [DOUBLE_UP, DOUBLE_DOWN]


# ==============================================================================
# Planar Hamiltonian Expressions
# ==============================================================================


class PlanarHamExpr(ExpressionBase):
    """Planar Hamiltonian Expression.

    Example:

    R = 2, C = 2 has a grid layout of R x (C + 1)
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

    def __init__(self, row: int, col: int) -> None:
        super().__init__(row=row, col=col)

    def __str__(self):
        return f"I({self.row}, {self.col})"


"""
    |p><p|
"""


class SingProj(PlanarHamExpr):
    """Single 2-local Hamiltonian."""

    def __init__(self, state: str, row: int, col: int) -> None:
        super().__init__(row, col)
        self.state = state

    def __str__(self) -> str:
        return f"P({self.state})_{self.row, self.col}"


"""
    |p_1 > < p_1|
    |p_2 > < p_2|
"""


class VertProj(PlanarHamExpr):
    """Vertical 2-local Hamiltonian."""

    def __init__(self, state1: str, state2: str, row: int, col: int) -> None:
        super().__init__(row, col)
        self.state1 = state1
        self.state2 = state2

    def __str__(self) -> str:
        return f"VP({self.state1}, {self.state2})_({self.row}, {self.col})"


"""
    |p_1 p_2> <p_1 p_2|
"""


class HoriProj(PlanarHamExpr):
    """Horizontal 2-local Hamiltonian."""

    def __init__(self, state1: str, state2: str, row: int, col: int) -> None:
        super().__init__(row, col)
        self.state1 = state1
        self.state2 = state2

    def __str__(self) -> str:
        return f"HP({self.state1}, {self.state2})_({self.row}, {self.col})"


"""
    |p_1 p_2><p_3 p_4| + |p_3 p_4><p_1 p_2|
"""


class HoriSymProject(PlanarHamExpr):
    """Horizontal 2-local Hamiltonian."""

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
    def __init__(self, n: int, R: int, H: PlanarHamExpr) -> None:
        self.grid: dict[tuple[int, int], list[PlanarHamExpr]] = {}
        for r in range(n):
            for c in range(R + 1):
                self.grid[r, c] = []

        self._gridify(H)

    # NOTE: the paper define n as 1-index, but R as 0-index
    def _gridify(self, H: PlanarHamExpr, scalar=1):
        if isinstance(H, PlanarHamExpr):
            if isinstance(H, ScalarSum):
                for H2 in H.Hs:
                    self._gridify(H2, scalar * H.scalar)
            else:
                H.scalar *= scalar
                print(H.scalar, "*", H)
                self.grid[H.row - 1, H.col].append(H)
        else:
            raise ValueError(f"Unexpected type {type(H)}")


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

"""
    q_2
    |
    | q_0
    |/ 
    q_1
"""


def encode3(s: str) -> str:
    """Particle to 3 qubits.

    Args:
        st (str): Particle State

    Returns:
        str: Encoding
    """
    if s is START:
        return "000"
    elif s is END:
        return "001"
    elif s is SINGLE_UP:
        return "010"
    elif s is SINGLE_DOWN:
        return "011"
    elif s is DOUBLE_UP:
        return "100"
    elif s is DOUBLE_DOWN:
        return "101"
    else:
        raise ValueError(f"{s} not supported ...")


_SPAN_1_FIRST = [int(encode3(x), 2) for x in SPAN_FIRST]
_SPAN_1_SECOND = [int(encode3(x), 2) for x in SPAN_SECOND]
_SPAN_2_FIRST = [
    int(encode3(x) + encode3(y), 2) for x in SPAN_FIRST for y in SPAN_FIRST
]
_SPAN_2_SECOND = [
    int(encode3(x) + encode3(y), 2) for x in SPAN_SECOND for y in SPAN_SECOND
]

# print(_SPAN_1_FIRST)
# print(_SPAN_1_SECOND)
# print(_SPAN_2_FIRST)
# print(_SPAN_2_SECOND)


def bin_strs(n: int) -> list[str]:
    return ["".join(p) for p in product("01", repeat=n)]


def kron_I(mat: sp.lil_matrix, front: int, back: int) -> sp.lil_matrix:
    assert front >= 0 and back >= 0, "Unexpected negative identity numbers"
    return sp.kron(sp.eye(2**front), sp.kron(mat, sp.eye(2**back)))


def init_square_matrix(size: int) -> sp.lil_matrix:
    return sp.lil_matrix((size, size)) * 1j


# the entry of compile
# different from compile_expr in regular clock
def reify3(n: int, R: int, H: PlanarHamExpr):
    num_particles = n * (R + 1)
    num_qubits = 3 * num_particles

    hori_all_qubits = 3 * (n + 1)
    hori_middle_qubits = 3 * (n - 1)

    size = 2**num_qubits
    size_hori_all = 2**hori_all_qubits

    my_H = init_square_matrix(size)
    for pos, Hs in Grid(n, R, H).grid.items():
        if len(Hs) == 0:
            continue

        start_idx = 3 * (pos[0] + pos[1] * n)
        for H in Hs:
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
                    for i1, i2 in enumerate(_SPAN_1_FIRST):
                        for j1, j2 in enumerate(_SPAN_1_SECOND):
                            X[i2, j2] = U[i1, j1]
                    # print(X.A.real)
                    X += X.conj().transpose()
                    my_H += H.scalar * kron_I(X, start_idx, num_qubits - start_idx - 3)

                elif U.shape[0] == 4:
                    X = init_square_matrix(64)
                    for i1, i2 in enumerate(_SPAN_2_FIRST):
                        for j1, j2 in enumerate(_SPAN_2_SECOND):
                            X[i2, j2] = U[i1, j1]
                    X += X.conj().transpose()
                    print(X)
                    my_H += H.scalar * kron_I(X, start_idx, num_qubits - start_idx - 6)

                else:
                    raise ValueError("Unexpected shape (should be 2x2 or 4x4)", U.shape)
            else:
                raise ValueError(f"{type(H)} not implemented")

    return my_H
