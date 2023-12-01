from typing import Literal

import numpy as np
import scipy.sparse as sp

from frontend import *


class ClockFrontend(Frontend):
    """Implement clock translation following:
    https://epubs.siam.org/doi/abs/10.1137/S0097539705447323
    """

    def __init__(self, locality: Literal["3", "5"]) -> None:
        """
        Args:
            locality (Literal[&quot;3&quot;, &quot;5&quot;]): K-locality of the generated Hamiltonian.
        """
        if locality not in ("3", "5"):
            raise ValueError(
                f"Expected locality to be 3 or 5 but obtained a locality of {locality} instead ..."
            )
        super().__init__()
        self.locality = locality

    def _gen_H_clock(self, n: int, L: int) -> HamExpr:
        """bottom of page 10, https://arxiv.org/pdf/quant-ph/0405098.pdf

        H_clock := Sum_{l=1}^{L-1} |01><01|_{l,l+1}^c,

        Args:
            n (int): number of computational qubits.
            L (int): number of gates.

        Returns:
            HamExpr: Hamiltonian that checks that the clock is prefixed by all 1's.
        """
        H_clock = Summation(
            [
                KronDiagonal(
                    Identity(l - 1),
                    KronDiagonal(
                        ProjectState("01"),
                        Identity(L - l - 1),
                    ),
                )
                for l in range(1, L)
            ]
        )

        # adjust for 3-local Hamiltonian
        if self.locality == "3":
            H_clock = ScalarMultiply(L**6, H_clock)

        return KronDiagonal(Identity(n), H_clock)

    def _gen_H_input(self, n: int, L: int) -> HamExpr:
        """top of page 11, https://arxiv.org/pdf/quant-ph/0405098.pdf

        H_input := (Sum_{i=1}^n |1><1|_i) ⨂ |0><0|_{1}^c

        Args:
            n (int): number of computational qubits.
            L (int): number of gates.

        Returns:
            HamExpr: Hamiltonian that checks that the input is all 0's.
        """
        return KronDiagonal(
            Summation(
                [
                    KronDiagonal(
                        Identity(i - 1),
                        KronDiagonal(
                            ProjectState("1"),
                            Identity(n - i),
                        ),
                    )
                    for i in range(1, n + 1)
                ]
            ),
            KronDiagonal(ProjectState("0"), Identity(L - 1)),
        )

    def _gen_H_clockinit(self, n: int, L: int) -> HamExpr:
        """top of page 11, https://arxiv.org/pdf/quant-ph/0405098.pdf

        H_clockinit := |1><1|_{1}^c

        Args:
            n (int): number of computational qubits.
            L (int): number of gates.

        Returns:
            HamExpr: Hamiltonian that checks that the clock is all 0's at start.
        """
        return KronDiagonal(
            Identity(n),
            KronDiagonal(ProjectState("1"), Identity(L - 1)),
        )

    def _gen_H_l_sum_part_check_clock(self, n: int, L: int) -> HamExpr:
        # when l = 1 or L (boundary cases)
        check_clock = [
            KronDiagonal(  # Eq 6: I ⨂ |00><00|_{1,2}^c
                ProjectState("00"), Identity(L - 2)
            ),
            KronDiagonal(  # Eq 6: I ⨂ |10><10|_{1,2}^c
                ProjectState("10"), Identity(L - 2)
            ),
            KronDiagonal(  # Eq 6: I ⨂ |10><10|_{L-1,L}^c
                Identity(L - 2), ProjectState("10")
            ),
            KronDiagonal(  # Eq 6: I ⨂ |11><11|_{L-1,L}^c
                Identity(L - 2), ProjectState("11")
            ),
        ]
        # when 2 <= l <= L-1
        if L > 2:
            clock_bwd, clock_fwd = [], []
            for l in range(2, L):
                clock_bwd.append(
                    KronDiagonal(  # Eq 5: I ⨂ |100><100|_{l-1,l,l+1}^c
                        Identity(l - 2),
                        KronDiagonal(ProjectState("100"), Identity(L - l - 1)),
                    )
                )
                clock_fwd.append(
                    KronDiagonal(  # Eq 5: I ⨂ |110><110|_{l-1,l,l+1}^c
                        Identity(l - 2),
                        KronDiagonal(ProjectState("110"), Identity(L - l - 1)),
                    )
                )

            check_clock.append(Summation(clock_bwd))
            check_clock.append(Summation(clock_fwd))

        return KronDiagonal(Identity(n), Summation(check_clock))

    def _gen_H_l_sum_part_unitary(self, L: int, Us: list[sp.spmatrix]) -> HamExpr:
        Us = [None] + Us  # For 1-indexing as in theorem

        encoded_unitaries = []

        if self.locality == "3":
            for l in range(1, L + 1):
                encoded_unitaries.append(EncodeUnitary(Us[l], l, L, "0", "1"))

        elif self.locality == "5":
            # Eq 6: U_1 ⨂ |10><00|_{1, 2}^c + U_1^\dagger ⨂ |00><10|_{1, 2}^c
            encoded_unitaries.append(EncodeUnitary(Us[1], 1, L, "00", "10"))
            # Eq 6: U_L ⨂ |11><10|_{L-1, L}^c + U_L^\dagger ⨂ |10><11|_{L-1, L}^c
            encoded_unitaries.append(EncodeUnitary(Us[L], L - 1, L, "10", "11"))
            # Eq 5: U_l ⨂ |110><100|_{l-1,l,l+1}^c + U_l^\dagger ⨂ |100><110|_{l-1,l,l+1}^c
            for l in range(2, L):
                encoded_unitaries.append(EncodeUnitary(Us[l], l - 1, L, "100", "110"))

        return Summation(encoded_unitaries)

    def _gen_H_l_sum(self, n: int, L: int, Us: list[sp.spmatrix]) -> HamExpr:
        """middle of page 11, https://arxiv.org/pdf/quant-ph/0405098.pdf

        Case l = 1:
              I ⨂ |00><00|_{1,2}^c
            - U_1 ⨂ |10><00|_{1, 2}^c
            - U_1^\dagger ⨂ |00><10|_{1, 2}^c
            + I ⨂ |10><10|_{1,2}^c

        Case 1 < l < L:
              I ⨂ |100><100|_{l-1,l,l+1}^c              (clock at beginning is valid)
            - U_l ⨂ |110><100|_{l-1,l,l+1}^c            (forward propagation is correct)
            - U_l^\dagger ⨂ |100><110|_{l-1,l,l+1}^c    (backward propagation is correct)
            + I ⨂ |110><110|_{l-1,l,l+1}^c              (clock at end is valid)

        Case l = L:
              I ⨂ |10><10|_{L-1,L}^c
            - U_L ⨂ |11><10|_{L-1, L}^c
            - U_L^\dagger ⨂ |10><11|_{L-1, L}^c
            + I ⨂ |11><11|_{L-1,L}^c

        Args:
            n (int): number of computational qubits.
            L (int): number of gates.
            Us (list[sp.spmatrix]): Unitaries.

        Returns:
            HamExpr: Hamlitonian that checks that the propagation of the unitaries is executed correctly.
        """
        # 1. Check the front and back clock propogated
        check_clock = self._gen_H_l_sum_part_check_clock(n, L)

        # 2. Encode unitary with the clock propagation
        encoded_unitaries = self._gen_H_l_sum_part_unitary(L, Us)

        # 3. Put everything together
        #   H_l_sum = 0.5 ((Sum_{l=1}^L Clock) - (Sum_{l=1}^L Propagation))
        return ScalarMultiply(
            0.5,
            Summation(
                [
                    check_clock,
                    ScalarMultiply(-1, encoded_unitaries),
                ]
            ),
        )

    def unitaries_to_program(self, Us: list[sp.spmatrix]):
        """Translate a list of unitaries into an adiabatic program.
        https://arxiv.org/pdf/quant-ph/0405098.pdf

        H_init   =  H_clockinit + H_input + H_clock
        H_final  =  H_propagate + H_input + H_clock

        Args:
            Us (list[sp.spmatrix]): A list of unitaries encoded as sp.sparse matrices.

        Returns:
            AdiabaticProgram: An adiabatic program containing an initial Hamiltonian
                and final Hamiltonian.
        """
        if len(Us) < 2:
            raise ValueError(
                f"Obtained unitary of length {len(Us)} when we require list of unitaries to be larger than 2 ..."
            )

        # Unpack state
        n = int(np.log2(Us[0].shape[0]))  # number of computational qubits.
        L = len(Us)  # number of gates -> clock qubits

        # 1. Compute H to check that the clock is well-formed.
        #    This is constant for H_init and H_final.
        H_clock = self._gen_H_clock(n, L)

        # 2. Compute H to check input.
        #    This is constant for H_init and H_final.
        H_input = self._gen_H_input(n, L)

        # 3. Compute H to check clock initialization.
        #    This is exclusive for H_init.
        H_clockinit = self._gen_H_clockinit(n, L)

        # 4. Compute H for step propagation.
        #    This is exclusive for H_final.
        H_l_sum = self._gen_H_l_sum(n, L, Us)

        # H_constant = Summation([H_input, KronDiagonal(Identity(n), H_clock)])
        H_constant = Summation([H_input, H_clock])

        return AdiabaticProgram(
            n,
            L,
            Summation([H_clockinit, H_constant]),
            Summation([H_l_sum, H_constant]),
            L ** (6 if self.locality == "5" else 10),
            L + 1,
        )
