import numpy as np

class TextBlock:
    def __init__(self, rows):
        assert isinstance(rows, list)
        self.rows = rows
        self.height = len(rows)
        self.width = max(map(len, rows))

    @classmethod
    def from_str(cls, data: str):
        assert isinstance(data, str)
        return cls(data.split('\n'))

    def format(self, width=None, height=None):
        width = width if width is not None else self.width
        height = height if height is not None else self.height
        lines = [f"{row:{width}}" for row in self.rows]
        lines += [' ' * width] * (height - self.height)
        return lines

    @staticmethod
    def merge(blocks):
        return [" ".join(parts) for parts in zip(*blocks)]

