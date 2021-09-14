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
__version__ = "0.2.2"

import numpy as _np
import numpy.core.getlimits as _np_getlimits

from . import _dd_ufunc
from . import _dd_linalg

ddouble = _dd_ufunc.dtype


def _register_finfo():
    DDouble = ddouble.type
    key = DDouble(-1).newbyteorder('<').tobytes()
    limits = _np_getlimits.MachArLike(ddouble,
                machep=-105, negep=-106, minexp=-1022, maxexp=1024, it=105,
                iexp=11, ibeta=2, irnd=5, ngrd=0, eps=_dd_ufunc.EPS,
                epsneg=_dd_ufunc.EPS/2, huge=_dd_ufunc.MAX,
                tiny=_dd_ufunc.MIN)
    params = dict(itype=_np.int64, fmt='%s', title="double-double number")
    _np_getlimits._register_type(limits, key)
    _np_getlimits._MACHAR_PARAMS[ddouble] = params
