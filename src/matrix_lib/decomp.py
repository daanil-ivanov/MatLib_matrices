import numpy as np
from .core import Matrix, FullMatrix, SymmetricMatrix

def lu(self: Matrix):
    m = self.flatten() if getattr(self, "flatten", None) and self.dtype == object else self
    n = m.height
    L = FullMatrix.zero(n,n,0.0)
    U = FullMatrix.zero(n,n,0.0)
    for i in range(n):
        L[i,i] = 1.0
        for j in range(i, n):
            U[i,j] = m[i,j] - np.sum(L[i,:i] * U[:i,j])
        for j in range(i+1, n):
            L[j,i] = (m[j,i] - np.sum(L[j,:i] * U[:i,i])) / U[i,i]
    return L, U

def det(self: Matrix):
    L, U = lu(self)
    d = 1
    for i in range(U.height):
        d *= U[i,i]
    return d

def pivot(self: FullMatrix):
    n = self.height
    P = FullMatrix.zero(n,n,0)
    P.data[:] = np.eye(n)
    A = FullMatrix(self.data.copy())
    for i in range(n):
        pivot = np.argmax(np.abs(A.data[i:,i])) + i
        if pivot != i:
            A.data[[i,pivot]] = A.data[[pivot,i]]
            P.data[[i,pivot]] = P.data[[pivot,i]]
    return A, P

def lup(self: FullMatrix):
    A, P = pivot(self)
    L, U = lu(A)
    return L, U, P

def ldl(self: FullMatrix):
    n = self.height
    L = FullMatrix.zero(n,n,0.0)
    D = [0]*n
    for i in range(n):
        L[i,i] = 1.0
        for j in range(i):
            s = sum(L[i,k]*D[k]*L[j,k] for k in range(j))
            L[i,j] = (self[i,j] - s) / D[j]
        D[i] = self[i,i] - sum(L[i,k]**2 * D[k] for k in range(i))
    return L, D

def sym_lu(self: SymmetricMatrix):
    L, D = ldl(self)
    n = self.height
    Dm = FullMatrix.zero(n,n,0.0)
    for i in range(n):
        Dm[i,i] = D[i]
    return L, FullMatrix(np.dot(Dm.data, L.data.T))

