import unittest

import numpy as np
import scipy.sparse as sp

from hamiltonian_lang import *

n = 4
exp2_n = 2**n

L = 4
exp2_L = 2**n


def assert_sp_matrix_equal(m1: sp.spmatrix, m2: sp.spmatrix) -> bool:
    assert np.allclose(m1.A, m2.A), "the input matrices are not equal"
