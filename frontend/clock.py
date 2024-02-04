from typing import Literal

import numpy as np

from frontend import *
from frontend.compress import *
from language.hamiltonian import *


class ClockAdiabaticProgram(AdiabaticProgram):
    """
    Args:
        num_data (int): The number of data qubits.
        num_clock (int): The number of clock qubits.
        ...: refer to the parent class
    """

    def __init__(
        self,
        num_data: int,
        num_clock: int,
        H_init: HamExpr,
        H_final: HamExpr,
        total_time: float,
        time_steps: int,
    ):
        super.__init__(H_init, H_final, total_time, time_steps, num_data + num_clock)
        self.num_data = num_data
        self.num_clock = num_clock

    def compile(self):
        return compile_expr(self.H_init), compile_expr(self.H_final)


class ClockFrontend(Frontend):
    """Implement clock translation from Section 3 and 4 from this paper:
    https://arxiv.org/abs/quant-ph/0405098.

    Args:
        locality (Literal["3", "5"]): The locality of the generated Hamiltonian.
    """

    def __init__(self, locality: Literal["3", "5"]) -> None:
        if locality not in ("3", "5"):
            raise ValueError(
                f"Expected locality to be 3 or 5 but obtained a locality of {locality} instead ..."
            )
        super().__init__()
        self.locality = locality

    def gen_H_clock(self, n: int, L: int) -> HamExpr:
        """H_clock, at the bottom of page 10.
        Adding energy to any two consective clock qubits that are "01".

        Args:
            n (int): number of data qubits.
            L (int): number of clock qubits.

        Returns:
            HamExpr: Hamiltonian that ensures legal clock states.
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

    def gen_H_input(self, n: int, L: int) -> HamExpr:
        """H_input, at the top of page 11.
        Adding energy to any data qubit that is "1" while the 1st clock qubit is "0".

        Args:
            n (int): number of data qubits.
            L (int): number of clock qubits.

        Returns:
            HamExpr: Hamiltonian that ensures that the input is all "0" when the 1st clock qubit as "0".
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

    def gen_H_clockinit(self, n: int, L: int) -> HamExpr:
        """H_clockinit, at the top of page 11.
        Adding energy to 1st clock qubit when it is "1".

        Args:
            n (int): number of data qubits.
            L (int): number of clock qubits.

        Returns:
            HamExpr: Hamiltonian that ensures that the 1st clock qubit is "0".
        """
        return KronDiagonal(
            Identity(n),
            KronDiagonal(ProjectState("1"), Identity(L - 1)),
        )

    def gen_H_l_sum_part_check_clock(self, n: int, L: int) -> HamExpr:
        """Partial sum of H_l, at the middle of page 11.
        Adding energy to each forward and backward clock state.

        Args:
            n (int): number of computational qubits.
            L (int): number of gates.

        Returns:
            HamExpr: Hamlitonian that checks the propagation of the clock part.
        """
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

    def gen_H_l_sum_part_unitary(self, L: int, Us: list[sp.spmatrix]) -> HamExpr:
        """Partial sum of H_l, at the middle of page 11.
        Reducing energy of each associated clock state with the unitaries.

        Args:
            n (int): number of computational qubits.
            L (int): number of gates.
            Us (list[sp.spmatrix]): List of unitaries.

        Returns:
            HamExpr: Hamlitonian that checks that the propagation of the unitaries.
        """
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

        return ScalarMultiply(-1, Summation(encoded_unitaries))

    def gen_H_l_sum(self, n: int, L: int, Us: list[sp.spmatrix]) -> HamExpr:
        """Sum of H_l, at the middle of page 11.
        Encoding unitary to corresponding clock state.

        Args:
            n (int): number of computational qubits.
            L (int): number of gates.
            Us (list[sp.spmatrix]): List of unitaries.

        Returns:
            HamExpr: Hamlitonian that checks that the propagation of the unitaries.
        """
        # 1. Check the front and back clock propogated
        check_clock = self.gen_H_l_sum_part_check_clock(n, L)

        # 2. Encode unitary with the clock propagation
        encoded_unitaries = self.gen_H_l_sum_part_unitary(L, Us)

        # 3. Put everything together
        return ScalarMultiply(0.5, Summation([check_clock, encoded_unitaries]))

    def unitaries_to_program(self, Us: list[sp.spmatrix]) -> ClockAdiabaticProgram:
        if len(Us) < 2:
            raise ValueError(
                f"Obtained unitary of length {len(Us)} when we require list of unitaries to be larger than 2 ..."
            )

        # Unpack state
        n = int(np.log2(Us[0].shape[0]))  # number of computational qubits.
        L = len(Us)  # number of gates -> clock qubits

        # 1. Compute H to check that the clock is well-formed.
        #    This is constant for H_init and H_final.
        H_clock = self.gen_H_clock(n, L)

        # 2. Compute H to check input.
        #    This is constant for H_init and H_final.
        H_input = self.gen_H_input(n, L)

        # 3. Compute H to check clock initialization.
        #    This is exclusive for H_init.
        H_clockinit = self.gen_H_clockinit(n, L)

        # 4. Compute H for step propagation.
        #    This is exclusive for H_final.
        H_l_sum = self.gen_H_l_sum(n, L, Us)

        # H_constant = Summation([H_input, KronDiagonal(Identity(n), H_clock)])
        H_constant = Summation([H_input, H_clock])

        return ClockAdiabaticProgram(
            n,
            L,
            Summation([H_clockinit, H_constant]),
            Summation([H_l_sum, H_constant]),
            L ** (6 if self.locality == "5" else 10),
            L + 1,
        )
