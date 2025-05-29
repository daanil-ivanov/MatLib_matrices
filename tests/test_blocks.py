import numpy as np
import pytest
from matrix_lib.blocks import TextBlock

def test_from_str_and_format():
    tb = TextBlock.from_str("a\nbb")
    assert tb.height == 2
    assert tb.width == 2
    lines = tb.format(width=3, height=3)
    assert len(lines) == 3
    assert lines[0].startswith("a  ")

