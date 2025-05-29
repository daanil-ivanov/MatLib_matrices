import numpy as np
import pytest
from matrix_lib.core import FullMatrix, Matrix

def test_zero_and_get_set():
    A = FullMatrix.zero(2,2,5)
    assert A.shape == (2,2)
    assert A[0,0] == 5
    A[1,1] = 7
    assert A[1,1] == 7

def test_add_sub_mul_repr():
    A = FullMatrix(np.array([[1,2],[3,4]], float))
    B = FullMatrix(np.array([[5,6],[7,8]], float))
    C = A + B
    assert np.array_equal(C.data, [[6,8],[10,12]])
    D = B - A
    assert np.array_equal(D.data, [[4,4],[4,4]])
    E = A @ B
    assert np.array_equal(E.data, [[19,22],[43,50]])
    assert ("|" in repr(A))

