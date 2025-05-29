import numpy as np
import pytest
from matrix_lib.core import FullMatrix
from matrix_lib.solve import check_trid, sweep, solve

def test_sweep_simple():
    A = FullMatrix(np.array([[3,1],[1,3]], float))
    b = FullMatrix(np.array([[5],[5]], float))
    assert check_trid(A)
    x = sweep(A, b)
    # правильный ответ — [1.25, 1.25]
    assert np.allclose(x.data.flatten(), [1.25, 1.25])

def test_solve_lu():
    A = FullMatrix(np.array([[3,1],[1,2]], float))
    b = FullMatrix(np.array([[5],[5]], float))
    x = solve(A, b)
    assert np.allclose(x.data.flatten(), [1,2])

