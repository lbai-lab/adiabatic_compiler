import numpy as np
import scipy.sparse as sp

from frontend import Frontend, PlanarAdiabaticProgram
from language.planar_hamiltonian import *

# Sunny's note, I use all 1-index to match paper
# they will be converted to 0 index in Grid class/method


class PlanarClockFrontend(Frontend):
    """Implement 2D clock translation following:
    https://epubs.siam.org/doi/abs/10.1137/S0097539705447323

    Example:

    n = 2, R = 2 has a grid layout of
        (r=0, c=0)  (r=0, c=1)  (r=0, c=2)
        (r=1, c=0)  (r=1, c=1)  (r=1, c=2)
        (r=2, c=0)  (r=2, c=1)  (r=2, c=2)
    """

    def __init__(self) -> None:
        """ """
        super().__init__()

    def _gen_H_input(self, n: int, R: int, L: int) -> PlanarHamExpr:
        """Check that input is not FIRST1.

        --------------
        | (r=1, c=0) |  (r=1, c=1)  (r=1, c=2)
        --------------
        | (r=2, c=0) |  (r=2, c=1)  (r=1, c=2)
        --------------
        | (r=3, c=0) |  (r=3, c=1)  (r=3, c=2)
        --------------

        Args:
            n (int): number of qubits (rows)
            R (int): number of rounds (columns without last)
            L (int): number of gates

        Returns:
            PlanarHamExpr: Hamiltonian.
        """
        return ScalarSum([SingProj(FIRST1, row=k, col=0) for k in range(1, n + 1)])

    def _gen_H_clockinit(self, n: int, R: int, L: int) -> PlanarHamExpr:
        """Check that clock is initialized correctly.

        --------------
        | (r=0, c=0) | (r=0, c=1)  (r=0, c=2)
        --------------
          (r=1, c=0)   (r=1, c=1)  (r=1, c=2)
          (r=2, c=0)   (r=2, c=1)  (r=2, c=2)

        Args:
            n (int): number of qubits (rows)
            R (int): number of rounds (columns without last)
            L (int): number of gates

        Returns:
            PlanarHamExpr: Hamiltonian.
        """
        return ScalarSum(
            [
                Identity(1, row=1, col=0),
                ScalarSum(
                    [
                        SingProj(FIRST0, row=1, col=0),
                        SingProj(FIRST1, row=1, col=0),
                    ],
                    scalar=-1,
                ),
            ]
        )

    def _gen_H_clock(self, n: int, R: int, L: int) -> PlanarHamExpr:
        """Check that clock is well-formed.

        Rules 1-4: for every two neighbor columns
        Rules 5-8:  for every two neighbor rows

        Args:
            n (int): number of qubits (rows)
            R (int): number of rounds (columns without last)
            L (int): number of gates

        Returns:
            PlanarHamExpr: Hamiltonian
        """
        rules = []

        # Horizontal Rules
        span_both = SPAN_FIRST + SPAN_SECOND
        not_start = span_both + [END]
        not_end = span_both + [START]
        for k in range(1, n + 1):  # all rows
            for r in range(R):  # one less neighbor column
                # RULE 1
                # START can't be at the left side of not-START
                for x in not_start:
                    rules.append(HoriProj(START, x, row=k, col=r))
                # RULE 2
                # END can't be at the right side of not-END
                for x in not_end:
                    rules.append(HoriProj(x, END, row=k, col=r))
                # RULE 3
                # START and END can't be adjacent
                rules.append(HoriProj(START, END, row=k, col=r))
                rules.append(HoriProj(END, START, row=k, col=r))
                # RULE 4
                # FIRST and SECOND can't be adjacent
                for x in span_both:
                    for y in span_both:
                        rules.append(HoriProj(x, y, row=k, col=r))

        # Vertical Rules
        not_second = SPAN_FIRST + [START, END]
        not_first = SPAN_SECOND + [START, END]
        for k in range(1, n):  # one less neighbor row
            for r in range(R + 1):  # all columns
                # RULE 5
                # SECOND can't be below not-SECOND
                for x in not_second:
                    for y in SPAN_SECOND:
                        rules.append(VertProj(x, y, row=k, col=r))
                # RULE 6
                # FIRST can't be above not-FIRST
                for x in not_first:
                    for y in SPAN_FIRST:
                        rules.append(VertProj(x, y, row=k, col=r))
                # RULE 7
                # START and END can't be adjacent
                rules.append(VertProj(START, END, row=k, col=r))
                rules.append(VertProj(END, START, row=k, col=r))
                # RULE 8
                # START can't be below SECOND
                for x in SPAN_SECOND:
                    rules.append(VertProj(x, START, row=k, col=r))
                # END can't be above FIRST
                for x in SPAN_FIRST:
                    rules.append(VertProj(END, x, row=k, col=r))

        return ScalarSum(rules, scalar=L**6)

    def _gen_H_l_sum(
        self, n: int, R: int, L: int, Us: list[sp.spmatrix]
    ) -> PlanarHamExpr:
        """Check that clock propagation is correct.

        Args:
            n (int): number of qubits (rows)
            R (int): number of rounds (columns without last)
            L (int): number of gates
            Us (_type_): list of Unitary matrices

        Returns:
            PlanarHamExpr: Hamiltonian.
        """
        # TODO check how unitary is appended, in 2x2/4x4 or 2^nx2^n form?
        arr_pos = []
        arr_neg = []
        Us = [None] + Us  # turn it into 1-index as paper

        for r in range(R):  # one less column (every neighbors are propogated)
            for k in range(1, n + 1):  # all rows
                # Downward Phase
                # check gates are propogated correctly
                down_l = 2 * n * r + k
                arr_pos.append(SymUnitary(Us[down_l], row=k, col=r))

                if k == 1:
                    for x in SPAN_FIRST:
                        arr_pos.append(SingProj(x, row=1, col=r))
                    for x in SPAN_SECOND:
                        for y in SPAN_FIRST:
                            arr_pos.append(VertProj(x, y, row=1, col=r))

                elif k == n:
                    for x in SPAN_SECOND:
                        for y in SPAN_FIRST:
                            arr_pos.append(VertProj(x, y, row=n - 1, col=r))
                    for x in SPAN_SECOND:
                        arr_pos.append(SingProj(x, row=n, col=r))

                else:
                    raise NotImplementedError(
                        "not tested with more than 2-qubit system"
                    )
                    for x in SPAN_SECOND:
                        for y in SPAN_FIRST:
                            arr_pos.append(VertProj(x, y, row=k - 1, col=r))
                            arr_pos.append(VertProj(x, y, row=k, col=r))

                # Upward Phase
                # applying identities from the bottom to the top
                i = n - k + 1
                if k == 1:
                    for x in SPAN_SECOND:
                        arr_pos.append(SingProj(x, row=n, col=r))
                    for x in SPAN_FIRST:
                        arr_pos.append(VertProj(START, x, row=n - 1, col=r + 1))

                elif k == n:
                    for x in SPAN_SECOND:
                        arr_pos.append(VertProj(x, END, row=1, col=r))
                    for x in SPAN_FIRST:
                        arr_pos.append(SingProj(x, row=1, col=r + 1))

                else:
                    raise NotImplementedError(
                        "not tested with more than 2-qubit system"
                    )
                    for x in SPAN_SECOND:
                        arr_pos.append(VertProj(x, END, row=i, col=r))
                    for x in SPAN_FIRST:
                        arr_pos.append(VertProj(START, x, row=i, col=r + 1))

                arr_neg.append(
                    HoriSymProject([SECOND0, START], [END, FIRST0], row=i, col=r)
                )
                arr_neg.append(
                    HoriSymProject([SECOND1, START], [END, FIRST1], row=i, col=r)
                )

        return ScalarSum(
            [ScalarSum(arr_pos), ScalarSum(arr_neg, scalar=-1)], scalar=0.5
        )

    def unitaries_to_program(self, Us: list[sp.spmatrix]):
        """Translate a list of unitaries into an adiabatic program.
        https://arxiv.org/pdf/quant-ph/0405098.pdf

        Args:
            Us (List[sp.spmatrix]): A list of unitaries encoded as sp.sparse matrices.

        Returns:
            an adiabatic program
        """
        if len(Us) < 4:
            raise ValueError(
                "Length of unitaries should be at least 4 (2-qubit circuit)"
            )

        L = len(Us)
        n = int(np.log2(Us[1].shape[0]))
        R = L // (2 * n)
        print(f"n={n}, L={L}, R={R}")

        H_input = self._gen_H_input(n, R, L)
        H_clockinit = self._gen_H_clockinit(n, R, L)
        H_clock = self._gen_H_clock(n, R, L)
        H_l_sum = self._gen_H_l_sum(n, R, L, Us)

        print(reify3(n, R, ScalarSum([H_clockinit])))

        return PlanarAdiabaticProgram(
            n,
            R,
            L,
            ScalarSum([H_clockinit, H_input, H_clock]),
            ScalarSum([H_l_sum, H_input, H_clock]),
            L**10,
            L + 1,
        )
