# Copyright (C) 2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
"""
Extension module for numpy providing the `ddouble` data type.

Loading this module registers an additional scalar data type `ddouble` with
numpy implementing double-double arithmetic.  You can use use the data type
by passing `dtype=xprec.ddouble` to numpy functions.

Example:

    import numpy as np
    from xprec import ddouble

    x = np.arange(5, dtype=ddouble)
    print(2 * x)

"""
__version__ = "1.0"

import numpy as _np

from . import _dd_ufunc

ddouble = _dd_ufunc.dtype


def finfo(dtype):
    dtype = _np.dtype(dtype)
    try:
        finfo_dunder = dtype.type.__finfo__
    except AttributeError:
        return _np.finfo(dtype)
    else:
        return finfo_dunder()
