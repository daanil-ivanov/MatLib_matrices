import numpy as np
from fractions import Fraction
from .blocks import TextBlock

class Matrix:
    @property
    def shape(self): raise NotImplementedError
    @property
    def dtype(self): raise NotImplementedError
    @property
    def width(self): return self.shape[1]
    @property
    def height(self): return self.shape[0]

    def __repr__(self):
        text = [[TextBlock.from_str(f"{self[r,c]}") for c in range(self.width)] 
                for r in range(self.height)]
        widths = [[el.width for el in row] for row in text]
        heights = [[el.height for el in row] for row in text]
        col_w = np.max(widths, axis=0)
        row_h = np.max(heights, axis=1)
        out = []
        for r in range(self.height):
            blocks = [ text[r][c].format(width=col_w[c], height=row_h[r]) 
                       for c in range(self.width) ]
            for line in TextBlock.merge(blocks):
                out.append(f"| {line} |")
            if r < self.height-1:
                sep = ' ' * (sum(col_w) + self.width)
                out.append(f"|{sep}|")
        return "\n".join(out)

    def empty_like(self, width=None, height=None): raise NotImplementedError
    def __getitem__(self, key): raise NotImplementedError
    def __setitem__(self, key, value): raise NotImplementedError

    def __add__(self, other):
        assert isinstance(other, Matrix) and self.shape==other.shape
        M = self.empty_like()
        for i in range(self.height):
            for j in range(self.width):
                M[i,j] = self[i,j] + other[i,j]
        return M

    def __sub__(self, other):
        assert isinstance(other, Matrix) and self.shape==other.shape
        M = self.empty_like()
        for i in range(self.height):
            for j in range(self.width):
                M[i,j] = self[i,j] - other[i,j]
        return M

    def __matmul__(self, other):
        assert isinstance(other, Matrix) and self.width==other.height
        M = self.empty_like(width=other.width, height=self.height)
        for i in range(self.height):
            for j in range(other.width):
                acc = None
                for k in range(self.width):
                    prod = self[i,k]*other[k,j]
                    acc = prod if acc is None else acc+prod
                M[i,j] = acc
        return M

    __mul__ = __matmul__

    def invert_element(self, element):
        if isinstance(element, (int, float, Fraction)):
            return 1/element
        if isinstance(element, Matrix):
            return element.inverse()
        raise TypeError

    def inverse(self): raise NotImplementedError


class FullMatrix(Matrix):
    def __init__(self, data: np.ndarray):
        assert isinstance(data, np.ndarray)
        self.data = data

    @property
    def shape(self): return self.data.shape
    @property
    def dtype(self): return self.data.dtype

    def empty_like(self, width=None, height=None):
        h = height or self.height
        w = width or self.width
        return FullMatrix(np.empty((h,w), dtype=self.data.dtype))

    @classmethod
    def zero(cls, height, width, default=0):
        arr = np.empty((height,width), dtype=type(default))
        arr[:] = default
        return cls(arr)

    def __getitem__(self, key): return self.data[key]
    def __setitem__(self, key, value): self.data[key] = value


class SymmetricMatrix(FullMatrix):
    @classmethod
    def zero(cls, dim, default=0):
        arr = np.full((dim,dim), default)
        return cls(arr)

    @property
    def dim(self): return self.height

    def __init__(self, data: np.ndarray):
        super().__init__(data)
        assert self.height==self.width
        assert np.allclose(self.data, self.data.T)

    def __setitem__(self, key, value):
        i,j = key
        self.data[i,j] = value
        self.data[j,i] = value


class BandMatrix(FullMatrix):
    def __init__(self, diag, upper, lower):
        import numpy as _np
        dim = len(diag)
        data = _np.zeros((dim,dim), dtype=object)
        super().__init__(data)
        for i in range(dim):
            self[i,i] = diag[i]
            for k, arr in enumerate(upper):
                if i < len(arr): self[i,i+k+1] = arr[i]
            for k, arr in enumerate(lower):
                if i < len(arr): self[i+k+1,i] = arr[i]

