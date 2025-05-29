from .blocks import TextBlock
from .core import Matrix, FullMatrix, SymmetricMatrix, BandMatrix
from .decomp import lu, lup, det, ldl, sym_lu
from .solve import check_trid, sweep, solve

# Экспортируем
__all__ = [
    "TextBlock",
    "Matrix", "FullMatrix", "SymmetricMatrix", "BandMatrix",
    "lu", "lup", "det", "ldl", "sym_lu",
    "check_trid", "sweep", "solve",
]

# Регистрируем методы в классах
Matrix.lu   = lu
Matrix.det  = det
FullMatrix.lup = lup
FullMatrix.ldl = ldl

# Метод solve уже прикреплён внутри solve.py: Matrix.solve = solve

