import scipy.sparse as sp
from typing import *


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

    def __init__(self, row=None, col=None, scalar=1) -> None:
        assert isinstance(row, int) or row is None
        assert isinstance(col, int) or col is None
        assert isinstance(scalar, int) or isinstance(scalar, float)
        self.row = row
        self.col = col
        self.scalar = scalar


class Identity(PlanarHamExpr):
    """Identity Operator.

    I
    """

    def __init__(self, n: int, row=None, col=None) -> None:
        super().__init__(row=row, col=col)
        self.n = n


class SingProj(PlanarHamExpr):
    """Single 2-local Hamiltonian.

    |p><p|
    """

    def __init__(self, state: str, row=None, col=None, scalar=1) -> None:
        super().__init__(row=row, col=col, scalar=scalar)
        self.state = state


class VertProj(PlanarHamExpr):
    """Vertical 2-local Hamiltonian.

    |p_1 > < p_1|
    |p_2 > < p_2|
    """

    def __init__(self, state1: str, state2: str, row=None, col=None, scalar=1) -> None:
        super().__init__(row=row, col=col, scalar=scalar)
        self.state1 = state1
        self.state2 = state2

    def __str__(self) -> str:
        if self.scalar == 1:
            return f"HP({self.state1}, {self.state2})_({self.row}, {self.col})"
        else:
            return f"{self.scalar} * VP({self.state1}, {self.state2})_({self.row}, {self.col})"


class HorizProj(PlanarHamExpr):
    """Horizontal 2-local Hamiltonian.

    |p_1 p_2> <p_1 p_2|
    """

    def __init__(self, state1: str, state2: str, row=None, col=None, scalar=1) -> None:
        super().__init__(row=row, col=col, scalar=scalar)
        self.state1 = state1
        self.state2 = state2

    def __str__(self) -> str:
        if self.scalar == 1:
            return f"HP({self.state1}, {self.state2})_({self.row}, {self.col})"
        else:
            return f"{self.scalar} * HP({self.state1}, {self.state2})_({self.row}, {self.col})"


class HorizSymProject(PlanarHamExpr):
    """Horizontal 2-local Hamiltonian.

    |p_1 p_2><p_3 p_4| + |p_3 p_4><p_1 p_2|
    """

    def __init__(
        self, states1: List[str], states2: List[str], row=None, col=None, scalar=1
    ) -> None:
        super().__init__(row=row, col=col, scalar=scalar)
        self.states1 = states1
        self.states2 = states2

    def __str__(self):
        if self.scalar == 1:
            return f"SP({self.orient}, {self.states1}, {self.states2})_({self.row}, {self.col})"
        else:
            return f"{self.scalar} * SP({self.orient}, {self.states1}, {self.states2})_({self.row}, {self.col})"


class VertSymUnitary(PlanarHamExpr):
    """Vertical 2-local Hamiltonian that symmetrizes unitary.

    0           U
    U^\dagger   0
    """

    def __init__(self, U: sp.spmatrix, row=None, col=None, scalar=1) -> None:
        super().__init__(row=row, col=col, scalar=scalar)
        self.U = U

    def __str__(self) -> str:
        if self.scalar == 1:
            return f"U_{self.row, self.col}"
        else:
            return f"{self.scalar} * U_{self.row, self.col}"


class Sum(PlanarHamExpr):
    """Sum of 2-local Hamiltonians."""

    def __init__(self, Hs: List[PlanarHamExpr], row=None, col=None, scalar=1) -> None:
        super().__init__(row=row, col=col, scalar=scalar)
        self.Hs = Hs

    def __str__(self) -> str:
        if self.scalar == 1:
            return "\n".join([str(H) for H in self.Hs])
        else:
            return str(self.scalar) + " * " + "\n".join([str(H) for H in self.Hs])


# ==============================================================================
# Grid
# ==============================================================================

HORIZ = "HORIZ"
VERT = "VERT"


class Grid:
    def __init__(self, rows: int, cols: int, H: PlanarHamExpr) -> None:
        self.rows = rows
        self.cols = cols
        self.H = H
        self.grid: dict[tuple[int, int], list[PlanarHamExpr]] = {}
        self._gridify()

    def _gridify(self):
        for r in range(1, self.rows + 1):
            for c in range(0, self.cols + 1):
                self.grid[r, c] = []

        def go(H: PlanarHamExpr):
            if isinstance(H, Identity):
                self.grid[H.row, H.col].append(H)
            elif isinstance(H, SingProj):
                self.grid[H.row, H.col].append(H)
            elif isinstance(H, HorizProj):
                self.grid[H.row, H.col].append(H)
            elif isinstance(H, VertProj):
                self.grid[H.row, H.col].append(H)
            elif isinstance(H, HorizSymProject):
                self.grid[H.row, H.col].append(H)
            elif isinstance(H, VertSymUnitary):
                self.grid[H.row, H.col].append(H)
            elif isinstance(H, Sum):
                for H2 in H.Hs:
                    go(H2)
            else:
                raise ValueError("Not supported ...")

        go(self.H)

    def filter(self, f: Callable) -> dict[tuple[int, int], PlanarHamExpr]:
        grid = {}
        for r in range(self.rows):
            for c in range(self.cols + 1):
                grid[(r, c)] = list(filter(f, self.grid[(r, c)]))
        return grid

    def zig_zag(self):
        vert = {}
        for r in range(self.rows):
            for c in range(self.cols + 1):
                vert[(r, c)] = []

        horiz = {}
        for r in range(self.rows):
            for c in range(self.cols + 1):
                horiz[(r, c)] = []

        for c in range(self.cols + 1):  # traverse columns
            for r in range(self.rows):  # traverse rows
                for H in self.grid[(r, c)]:
                    if isinstance(H, Identity):
                        vert[(r, c)].append(H)
                    elif isinstance(H, SingProj):
                        vert[(r, c)].append(H)
                    elif isinstance(H, VertProj):
                        vert[(r, c)].append(H)
                    elif isinstance(H, VertSymUnitary):
                        vert[(r, c)].append(H)
                    elif isinstance(H, HorizProj):
                        horiz[(r, c)].append(H)
                    elif isinstance(H, HorizSymProject):
                        horiz[(r, c)].append(H)
                    else:
                        raise ValueError(f"{type(H)} not expected ...")

        acc = []
        for c in range(self.cols + 1):
            for r in range(self.rows):
                acc.append((r, c, VERT, vert[(r, c)]))
            for r in range(self.rows - 1, -1, -1):
                acc.append((r, c, HORIZ, horiz[(r, c)]))
        return acc


# ==============================================================================
# Utility
# ==============================================================================


def is_vert(H: PlanarHamExpr, inc_sing=False) -> bool:
    if isinstance(H, Identity):
        return inc_sing
    elif isinstance(H, SingProj):
        return inc_sing
    elif isinstance(H, HorizProj):
        return False
    elif isinstance(H, VertProj):
        return True
    elif isinstance(H, HorizSymProject):
        return False
    elif isinstance(H, VertSymUnitary):
        return True
    elif isinstance(H, Sum):
        return False
    else:
        raise ValueError(f"{type(H)} not supported ...")


def is_horiz(H: PlanarHamExpr, inc_sing=False) -> bool:
    if isinstance(H, Identity):
        return inc_sing
    elif isinstance(H, SingProj):
        return inc_sing
    elif isinstance(H, HorizProj):
        return True
    elif isinstance(H, VertProj):
        return False
    elif isinstance(H, HorizSymProject):
        return True
    elif isinstance(H, VertSymUnitary):
        return False
    elif isinstance(H, Sum):
        return False
    else:
        raise ValueError(f"{type(H)} not supported ...")


# def gridify(rows: int, cols: int, H: PlanarHamExpr) -> dict[tuple[int, int], PlanarHamExpr]:
#     return Grid(rows, cols, H).grid


# def gridify_vert(
#     R: int, C: int, H: PlanarHamExpr
# ) -> dict[tuple[int, int], PlanarHamExpr]:
#     return Grid(R, C, H).filter(lambda x: is_vert(x, inc_sing=False))


# def gridify_horiz(
#     R: int, C: int, H: PlanarHamExpr
# ) -> dict[tuple[int, int], PlanarHamExpr]:
#     return Grid(R, C, H).filter(lambda x: is_horiz(x, inc_sing=False))


#
#  FROM refiy.py
#
import numpy as np
import scipy.sparse as sp
from typing import *

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


def bin_strs(n: int) -> list[str]:
    from itertools import product

    return ["".join(p) for p in product("01", repeat=n)]


def flatten3(n: int, R: int, i: int, j: int, q: int) -> int:
    return q + 3 * i + j * 3 * n


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


_START = encode3(START)
_FIRST0 = encode3(FIRST0)
_FIRST1 = encode3(FIRST1)
_SECOND0 = encode3(SECOND0)
_SECOND1 = encode3(SECOND1)
_END = encode3(END)
_ILLEGAL0 = "110"
_ILLEGAL1 = "111"


def to_vec(s):
    def f(c):
        if c == "0":
            return np.array([1.0, 0.0])
        else:
            return np.array([0.0, 1.0])

    if len(s) == 1:
        return f(s)
    else:
        return np.kron(f(s[0]), to_vec(s[1:]))


# ==============================================================================
# Reify
# ==============================================================================


def penalize_11(X):
    illegal = [_ILLEGAL0, _ILLEGAL1]
    for n in bin_strs(3):
        for s in illegal:
            idx = int(s + n, 2)
            X[idx, idx] += 1.0
            idx = int(n + s, 2)
            X[idx, idx] += 1.0


def reify_2local(H: PlanarHamExpr):
    if isinstance(H, Identity):
        return H.scalar * np.eye(64) * 1j
    elif isinstance(H, Sum):
        return sum(reify_2local(x) for x in H.Hs)
    elif isinstance(H, SingProj):
        X = sp.lil_matrix((8, 8)) * 1j
        s = encode3(H.state)
        idx = int(s, 2)
        X[idx, idx] = 1.0
        Y = sp.kron(sp.eye(8), X)  # |bot top>
        return H.scalar * Y
    elif isinstance(H, VertProj):
        X = sp.lil_matrix((64, 64)) * 1j
        s1 = encode3(H.state1)
        s2 = encode3(H.state2)
        idx = int(s2 + s1, 2)  # |bot top>
        X[idx, idx] = 1.0
        return H.scalar * X
    elif isinstance(H, VertSymUnitary):
        U = H.U
        if U.shape[0] == 2:
            X = sp.lil_matrix((8, 8)) * 1j
            X[int(_FIRST0, 2), int(_SECOND0, 2)] = U[0, 0]
            X[int(_FIRST0, 2), int(_SECOND1, 2)] = U[0, 1]
            X[int(_FIRST1, 2), int(_SECOND0, 2)] = U[1, 0]
            X[int(_FIRST1, 2), int(_SECOND1, 2)] = U[1, 1]
            X += X.conj().transpose()
            X = -X
            # |bot top>
            return H.scalar * sp.kron(sp.eye(8), X)
        elif U.shape[0] == 4:
            X = sp.lil_matrix((64, 64)) * 1j
            # |bot top>
            # 00 -> [00, 01, 10, 11]
            X[int("010100", 2), int("100100", 2)] = U[0, 0]
            X[int("010100", 2), int("100101", 2)] = U[0, 1]
            X[int("010100", 2), int("101100", 2)] = U[0, 2]
            X[int("010100", 2), int("101101", 2)] = U[0, 3]
            # 01 -> [00, 01, 10, 11]
            X[int("010101", 2), int("100100", 2)] = U[1, 0]
            X[int("010101", 2), int("100101", 2)] = U[1, 1]
            X[int("010101", 2), int("101100", 2)] = U[1, 2]
            X[int("010101", 2), int("101101", 2)] = U[1, 3]
            # 10 -> [00, 01, 10, 11]
            X[int("011100", 2), int("100100", 2)] = U[2, 0]
            X[int("011100", 2), int("100101", 2)] = U[2, 1]
            X[int("011100", 2), int("101100", 2)] = U[2, 2]
            X[int("011100", 2), int("101101", 2)] = U[2, 3]
            # 11 -> [00, 01, 10, 11]
            X[int("011101", 2), int("100100", 2)] = U[3, 0]
            X[int("011101", 2), int("100101", 2)] = U[3, 1]
            X[int("011101", 2), int("101100", 2)] = U[3, 2]
            X[int("011101", 2), int("101101", 2)] = U[3, 3]
            X += X.conj().transpose()
            X = -X
            return H.scalar * X
        else:
            raise ValueError(f"Shape {U.shape} not expected ...")
    elif isinstance(H, HorizProj):
        X = sp.lil_matrix((64, 64)) * 1j
        s1 = encode3(H.state1)
        s2 = encode3(H.state2)
        idx = int(s2 + s1, 2)  # |right left>
        X[idx, idx] = 1.0
        return H.scalar * X
    elif isinstance(H, HorizSymProject):
        X = sp.lil_matrix((64, 64)) * 1j
        s1_1 = encode3(H.states1[0])
        s1_2 = encode3(H.states1[1])
        s2_1 = encode3(H.states2[0])
        s2_2 = encode3(H.states2[1])
        idx1 = int(s1_2 + s1_1, 2)  # |right left>
        idx2 = int(s2_2 + s2_1, 2)  # |right left>
        X[idx1, idx2] = 1.0
        X[idx2, idx1] = 1.0
        return H.scalar * X
    else:
        raise ValueError(f"{type(H)} not expected ...")


def reify_zigzag(grid: Grid):
    acc = []
    for r, c, orient, Hs in grid.zig_zag():
        my_H = np.zeros((64, 64))
        for H in Hs:
            my_H += reify_2local(H)
        acc.append((r, c, orient, my_H))
    return acc


# ==============================================================================
# Global Reify (WARNING: CAN ONLY HANDLE SMALL SYSTEMS)
# ==============================================================================


def reify3(n: int, R: int, H: PlanarHamExpr):
    grid = Grid(n, R, H).grid
    num_qubits = 3 * n * (R + 1)
    size = 2**num_qubits
    my_H = sp.lil_matrix((size, size))

    def splat(i, j):
        q0 = flatten3(n, R, i, j, 0)
        q1 = flatten3(n, R, i, j, 1)
        q2 = flatten3(n, R, i, j, 2)
        return [q0, q1, q2]

    for i in range(n):
        for j in range(R + 1):
            # print("HERE", i, j, len(grid[i, j]))
            for H in grid[i, j]:
                if isinstance(H, Identity):
                    my_H += H.scalar * sp.eye(size)
                elif isinstance(H, SingProj):
                    D = 2 ** (3)
                    X = sp.lil_matrix((D, D)) * 1j
                    s, qs = encode3(H.state), splat(i, j)
                    idx = int(s, 2)
                    X[idx, idx] = 1.0
                    my_H += H.scalar * sp.kron(
                        sp.eye(2 ** (num_qubits - qs[-1] - 1)),
                        sp.kron(X, sp.eye(2 ** qs[0])),
                    )
                elif isinstance(H, VertProj):
                    D = 2 ** (3 * 2)
                    X = sp.lil_matrix((D, D)) * 1j
                    s1, qs1 = encode3(H.state1), splat(i, j)
                    s2, qs2 = encode3(H.state2), splat(i + 1, j)
                    idx = int(s1 + s2, 2)
                    X[idx, idx] = 1.0
                    my_H += H.scalar * sp.kron(
                        sp.eye(2 ** (num_qubits - qs2[-1] - 1)),
                        sp.kron(X, sp.eye(2 ** qs1[0])),
                    )
                elif isinstance(H, HorizProj):
                    s1, qs1 = encode3(H.state1), splat(i, j)
                    s2, qs2 = encode3(H.state2), splat(i, j + 1)
                    D = 2 ** (qs2[-1] - qs1[0] + 1)
                    X = sp.lil_matrix((D, D)) * 1j
                    for middle in bin_strs(qs2[0] - qs1[-1] - 1):
                        idx = int(s1 + middle + s2, 2)
                        X[idx, idx] = 1.0
                    # print(i, j)
                    my_H += H.scalar * sp.kron(
                        sp.eye(2 ** (num_qubits - qs2[-1] - 1)),
                        sp.kron(X, sp.eye(2 ** qs1[0])),
                    )
                elif isinstance(H, HorizSymProject):
                    # print("SYMPROJECT", i, j)

                    qs = []
                    for k in range(len(H.states1)):
                        q0 = flatten3(n, R, i, j + k, 0)
                        q1 = flatten3(n, R, i, j + k, 1)
                        q2 = flatten3(n, R, i, j + k, 2)
                        qs.append(q0, q1, q2)
                    D = 2 ** (qs[-1] - qs[0] + 1)
                    X = sp.lil_matrix((D, D)) * 1j
                    s1_1 = encode3(H.states1[0])
                    s1_2 = encode3(H.states1[1])
                    s2_1 = encode3(H.states2[0])
                    s2_2 = encode3(H.states2[1])
                    for middle in bin_strs(qs[3] - qs[2] - 1):
                        idx1 = int(s1_1 + middle + s1_2, 2)
                        idx2 = int(s2_1 + middle + s2_2, 2)
                        X[idx1, idx2] = 1.0
                    my_H += H.scalar * sp.kron(
                        sp.eye(2 ** (num_qubits - qs[-1] - 1)),
                        sp.kron(X, sp.eye(2 ** qs[0])),
                    )
                    # print("HORIZONTAL", H.row, H.col, len(H.states1), qs)
                elif isinstance(H, VertSymUnitary):
                    # print("SYMUNITARY", i, j)
                    D = 2 ** (3 * 2)
                    U = H.U
                    if U.shape[0] == 2:
                        X = sp.lil_matrix((8, 8)) * 1j
                        X[int("010", 2), int("100", 2)] = U[0, 0]
                        X[int("010", 2), int("101", 2)] = U[0, 1]
                        X[int("011", 2), int("100", 2)] = U[1, 0]
                        X[int("011", 2), int("101", 2)] = U[1, 1]
                        X += X.conj().transpose()
                        X = -X
                        # print("TEST X", np.allclose(X.todense(), X.conj().transpose().todense()))
                        qs = splat(i, j)
                        # Y = H.scalar * sp.kron(sp.eye(2 ** (num_qubits - qs[-1] - 1)), sp.kron(X, sp.eye(2 ** qs[0])))
                        # print("TEST Y", np.allclose(Y.todense(), Y.conj().transpose().todense()))
                        my_H += H.scalar * sp.kron(
                            sp.eye(2 ** (num_qubits - qs[-1] - 1)),
                            sp.kron(X, sp.eye(2 ** qs[0])),
                        )
                    elif U.shape[0] == 4:
                        # print("HERE", U)
                        X = sp.lil_matrix((64, 64)) * 1j
                        # |bot top>
                        # 00 -> [00, 01, 10, 11]
                        X[int("010100", 2), int("100100", 2)] = U[0, 0]
                        X[int("010100", 2), int("100101", 2)] = U[0, 1]
                        X[int("010100", 2), int("101100", 2)] = U[0, 2]
                        X[int("010100", 2), int("101101", 2)] = U[0, 3]
                        # 01 -> [00, 01, 10, 11]
                        X[int("010101", 2), int("100100", 2)] = U[1, 0]
                        X[int("010101", 2), int("100101", 2)] = U[1, 1]
                        X[int("010101", 2), int("101100", 2)] = U[1, 2]
                        X[int("010101", 2), int("101101", 2)] = U[1, 3]
                        # 10 -> [00, 01, 10, 11]
                        X[int("011100", 2), int("100100", 2)] = U[2, 0]
                        X[int("011100", 2), int("100101", 2)] = U[2, 1]
                        X[int("011100", 2), int("101100", 2)] = U[2, 2]
                        X[int("011100", 2), int("101101", 2)] = U[2, 3]
                        # 11 -> [00, 01, 10, 11]
                        X[int("011101", 2), int("100100", 2)] = U[3, 0]
                        X[int("011101", 2), int("100101", 2)] = U[3, 1]
                        X[int("011101", 2), int("101100", 2)] = U[3, 2]
                        X[int("011101", 2), int("101101", 2)] = U[3, 3]
                        X += X.conj().transpose()
                        X = -X
                        # print("TEST X 2", np.allclose(X.todense(), X.conj().transpose().todense()))
                        qs1, qs2 = splat(i, j), splat(i + 1, j)
                        my_H += H.scalar * sp.kron(
                            sp.eye(2 ** (num_qubits - qs2[-1] - 1)),
                            sp.kron(X, sp.eye(2 ** qs1[0])),
                        )
                    else:
                        raise ValueError(
                            "Unexpected shape (should be 2x2 or 4x4)", U.shape
                        )
                else:
                    raise ValueError(f"{type(H)} not implemented")
    return my_H


# ==============================================================================
# Test
# ==============================================================================


def mk_oneq_H(U):
    X = np.zeros((8, 8)) * 1j
    X[int(_FIRST0, 2), int(_SECOND0, 2)] = U[0, 0]
    X[int(_FIRST0, 2), int(_SECOND1, 2)] = U[0, 1]
    X[int(_FIRST1, 2), int(_SECOND0, 2)] = U[1, 0]
    X[int(_FIRST1, 2), int(_SECOND1, 2)] = U[1, 1]
    X += X.conj().transpose()
    X = -X

    # Penalize 11 prefix
    illegal = [_ILLEGAL0, _ILLEGAL1]
    for s in illegal:
        idx = int(s, 2)
        X[idx, idx] += 1.0

    # Penalize non-input
    for i in range(8):
        X[i, i] += 1.0
    X[int("010", 2), int("010", 2)] -= 1
    return X


def mk_twoq_H(U):
    X = np.array(reify_2local(VertSymUnitary(U, 0, 0)).todense())

    # Penalize 11 prefix
    penalize_11(X)

    # Penalize non-input
    for i in range(64):
        X[i, i] += 1.0
    # X[int("010010", 2), int("010010", 2)] -= 1
    # second phase 0 on top of first phase 0
    X[int("010100", 2), int("010100", 2)] -= 1

    # Penalize first above second
    for x in [_FIRST0, _FIRST1]:
        for y in [_SECOND0, _SECOND1]:
            X[int(x + y, 2), int(x + y, 2)] += 1
    return X


def test_oneq(U):
    X = mk_oneq_H(U)

    eigvals, eigvecs = np.linalg.eigh(X)
    print(eigvals)
    for i in range(1):  # for i in range(8):
        x = np.abs(eigvecs[:, i])
        args = np.argsort(x)
        acc = []
        for j in range(1, 5):
            s = format(args[-j], "03b")
            acc.append((s, x[args[-j]]))
        print(acc)


def test_twoq(U):
    X = mk_twoq_H(U)

    eigvals, eigvecs = np.linalg.eigh(X)
    print(eigvals[0:64])
    for i in range(5):  # for i in range(8):
        x = np.abs(eigvecs[:, i])
        args = np.argsort(x)
        acc = []
        for j in range(1, 4):
            s = format(args[-j], "06b")
            acc.append((s, x[args[-j]]))
        print(acc)


def test():
    H = (
        1
        / np.sqrt(2)
        * np.array(
            [
                [1, 1],
                [1, -1],
            ]
        )
    )
    print("H")
    test_oneq(H)

    theta = np.pi
    X = np.array(
        [
            [np.cos(theta / 2), np.sin(theta / 2) * -1j],
            [np.sin(theta / 2) * -1j, np.cos(theta / 2)],
        ]
    )
    print("X")
    test_oneq(X)

    print("X I")  # 2 particles
    test_twoq(np.kron(X, np.eye(2)))

    print("X H")  # 2 particles
    test_twoq(np.kron(X, H))


if __name__ == "__main__":
    test()
