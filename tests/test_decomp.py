import numpy as np
import pytest
from matrix_lib.core import FullMatrix
from matrix_lib.decomp import lu, det, lup, ldl, sym_lu

def test_lu_and_det():
    A = FullMatrix(np.array([[2,0,0],[1,3,4],[0,5,6]], float))
    L, U = lu(A)
    R = L @ U
    assert np.allclose(R.data, A.data)
    assert det(A) == pytest.approx(np.linalg.det(A.data))

def test_lup_pivot():
    A = FullMatrix(np.array([[0,2],[1,3]], float))
    L, U, P = lup(A)
    assert np.allclose((P @ A).data, (L @ U).data)

def test_ldl_and_sym_lu():
    data = np.array([[4,2],[2,3]], float)
    S = FullMatrix(data)
    L, D = ldl(S)
    Dt = np.diag(D)
    assert np.allclose(L.data @ Dt @ L.data.T, data)
    L2, U2 = sym_lu(S)
    assert np.allclose((L2 @ U2).data, data)

