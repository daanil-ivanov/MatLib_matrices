import numpy as np
from .core import Matrix, FullMatrix

def check_trid(self: FullMatrix):
    n = self.height
    assert n == self.width
    for i in range(n):
        for j in range(i+2, n):
            if self[i,j] != 0 or self[j,i] != 0:
                return False
    return True

def sweep(self: FullMatrix, d: FullMatrix):
    assert check_trid(self)
    A = FullMatrix(self.data.copy())
    B = FullMatrix(d.data.copy())
    n = A.height
    # прямой ход
    for i in range(1,n):
        w = A[i,i-1] / A[i-1,i-1]
        A[i,i] -= w * A[i-1,i]
        B[i,0] -= w * B[i-1,0]
    # обратный ход
    X = FullMatrix.zero(n,1,0.0)
    X[n-1,0] = B[n-1,0] / A[n-1,n-1]
    for i in range(n-2, -1, -1):
        X[i,0] = (B[i,0] - A[i,i+1]*X[i+1,0]) / A[i,i]
    return X

def solve(self: Matrix, d: FullMatrix):
    L, U = self.lu()
    n = L.height
    y = FullMatrix.zero(n,1,0.0)
    for i in range(n):
        y[i,0] = d[i,0] - np.dot(L.data[i,:i], y.data[:i,0])
    x = FullMatrix.zero(n,1,0.0)
    for i in range(n-1, -1, -1):
        x[i,0] = (y[i,0] - np.dot(U.data[i,i+1:], x.data[i+1:,0])) / U.data[i,i]
    return x

# Регистрируем метод solve в Matrix
Matrix.solve = solve

