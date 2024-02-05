from language import *

"""[Note] A language for expressing Hamiltonian matrices.

Hermitian matrices
    H ::= D | U(i1, i2, state_bwd, state_fwd) | c * H | Σ H

Real-valued diagonal matrices
    D ::= I^{⨂n} | |x><x| | D ⨂ D

"""


# ==============================================================================
# Hamiltonian Expressions
# ==============================================================================


class HamExpr(ExpressionBase):
    """
    The base expression for this language.
    The matrix represented by this category of expression is always Hermetian.
    """

    pass


class EncodeUnitary(HamExpr):
    """
    Encode a unitary as a hamiltonian with forward and backward state.

    Args:
        U (sp.spmatrix): Unitary.
        start (int): start position (1-indexing).
        bound (int): number of clock qubits, which is L.
        state_bwd (str): backward clock.
        state_fwd (str): forward clock.
    """

    def __init__(
        self,
        U: sp.spmatrix,
        start: int,
        bound: int,
        state_bwd: str,
        state_fwd: str,
    ) -> None:
        assert isinstance(U, sp.spmatrix)
        assert isinstance(start, int)
        assert isinstance(bound, int)
        assert isinstance(state_bwd, str)
        assert isinstance(state_fwd, str)
        assert len(state_bwd) == len(state_fwd)
        assert start >= 1
        assert bound >= start - 1 + len(state_fwd)

        self.U = U
        self.front_i = start - 1
        self.back_i = bound - len(state_bwd) - self.front_i
        self.state_fwd = state_fwd
        self.state_bwd = state_bwd

    def __str__(self) -> str:
        return f"U({str(self.U.todense())}, {self.state_bwd} -> {self.state_fwd}, clock {self.front_i} to {self.back_i})"


class ScalarMultiply(HamExpr):
    """
    Scalar multiple of a Hamiltonian.

    Args:
        scalar (int | float): Scalar value.
        expr (HamExpr): Hamiltonian.
    """

    def __init__(self, scalar: int | float, expr: HamExpr) -> None:
        assert isinstance(scalar, (int, float))
        assert isinstance(expr, HamExpr)

        self.scalar = scalar
        self.expr = expr

    def __str__(self) -> str:
        return f"({str(self.scalar)} * {str(self.expr)})"


class Summation(HamExpr):
    """
    Sum of Hamiltonians.

    Args:
        expr_arr (list[HamExpr]): Hamiltonians to sum up.
    """

    def __init__(self, expr_arr: list[HamExpr]) -> None:
        assert len(expr_arr) > 0
        assert all(isinstance(x, HamExpr) for x in expr_arr)

        self.expr_arr = expr_arr

    def __str__(self) -> str:
        return "(" + "\n+ ".join([str(e) for e in self.expr_arr]) + ")"


# ---------------------------------------------------------
# Diagonal Expressions
# ---------------------------------------------------------


class Diagonal(HamExpr):
    """
    Real-Valued Diagonal.
    """

    def _is_diag(expr: HamExpr) -> bool:
        """
        Check to see if the input Hamiltonian expression is diagonal.

        Args:
            expr (HamExpr): Hamiltonian expression.

        Returns:
            bool: Whether the expression is diagonal.
        """
        if isinstance(expr, Diagonal):
            return True
        elif isinstance(expr, Summation):
            for x in expr.expr_arr:
                if not KronDiagonal._is_diag(x):
                    return False
            return True
        elif isinstance(expr, ScalarMultiply):
            return KronDiagonal._is_diag(expr.expr)
        else:
            return False


class Identity(Diagonal):
    """
    Identity matrix over n qubits.

    Args:
        n (int): number of qubits.
    """

    def __init__(self, n: int) -> None:
        assert isinstance(n, int)

        self.n = n

    def __str__(self) -> str:
        return f"I({self.n})"


class ProjectState(Diagonal):
    """
    Project a state to form as a Hamiltonian.

    Args:
        state (str): state to project onto.
    """

    def __init__(self, state: str) -> None:
        assert isinstance(state, str)

        self.state = state

    def __str__(self) -> str:
        return f"(|{self.state}><{self.state}|)"


class KronDiagonal(Diagonal):
    """
    Kronocker product of 2 real-valued diagonal matrices.

    Args:
        D1 (Diagonal): First real-valued diagonal matrix.
        D2 (Diagonal): Second real-valued diagonal matrix.
    """

    def __init__(self, D1: Diagonal, D2: Diagonal) -> None:
        assert Diagonal._is_diag(D1)
        assert Diagonal._is_diag(D2)

        self.D1 = D1
        self.D2 = D2

    def __str__(self) -> str:
        return f"({str(self.D1)} ⨂ {str(self.D2)})"


# ==============================================================================
# Compilation
# ==============================================================================


def compile_expr(expr: HamExpr) -> sp.spmatrix:
    """
    Compute the Hamiltonian expression into a scipy matrix.

    Args:
        expr (HamExpr): Expression to lower.

    Returns:
        sp.spmatrix: actual matrix of this Hamiltonian.
    """
    if isinstance(expr, Identity):
        return sp.eye(2**expr.n)
    elif isinstance(expr, ProjectState):
        size = 2 ** len(expr.state)
        zeros = sp.lil_matrix((size, size))
        idx = int(expr.state, 2)
        zeros[idx, idx] = 1
        return zeros
    elif isinstance(expr, KronDiagonal):
        return sp.kron(compile_expr(expr.D1), compile_expr(expr.D2))
    elif isinstance(expr, Summation):
        return sum(compile_expr(x) for x in expr.expr_arr)
    elif isinstance(expr, ScalarMultiply):
        return expr.scalar * compile_expr(expr.expr)
    elif isinstance(expr, EncodeUnitary):
        size = 2 ** len(expr.state_bwd)
        zeros = sp.lil_matrix((size, size))
        zeros[int(expr.state_fwd, 2), int(expr.state_bwd, 2)] = 1
        res = sp.kron(sp.kron(expr.U, sp.eye(2**expr.front_i)), zeros)
        return sp.kron(res + res.conj().transpose(), sp.eye(2**expr.back_i))
    else:
        raise ValueError(f"Shouldn't happen ... {type(expr)}")
