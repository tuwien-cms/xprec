# Copyright (C) 2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from xprec import ddouble


COMPATIBLE_DTYPES = [
    np.int8, np.int16, np.int32, np.int64, np.bool_, np.float32, np.float64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    ]


@pytest.mark.parametrize('other', COMPATIBLE_DTYPES)
def test_cast_from(other):
    assert np.can_cast(other, ddouble, 'unsafe')
    assert np.can_cast(other, ddouble, 'safe')

    x = np.eye(3, dtype=other)
    y = x.astype(ddouble)
    assert (x == y).all()


@pytest.mark.parametrize('other', COMPATIBLE_DTYPES)
def test_cast_to(other):
    assert np.can_cast(ddouble, other, 'unsafe')
    assert not np.can_cast(ddouble, other, 'safe')

    x = np.eye(3, dtype=ddouble)
    y = x.astype(other)
    assert (x == y).all()
