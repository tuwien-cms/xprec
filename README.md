Library for double-double arithmetic calculation
================================================

Extension module for numpy providing the `ddouble` data type.

Loading this module registers an additional scalar data type `ddouble` with
numpy implementing double-double arithmetic.  You can use use the data type
by passing `dtype=xprec.ddouble` to numpy functions.

The `xprec.linalg` module provides some linear algebra subroutines, in
particular QR, RRQR, SVD and truncated SVD.

Installation
------------

    $ pip install xprec

Quickstart
----------

    import numpy as np
    x = np.linspace(0, np.pi)

    # import double-double precision data type
    from xprec import ddouble
    x = x.astype(ddouble)
    y = x * x + 1
    z = np.sin(x)

    # do some linalg
    import xprec.linalg
    A = np.vander(np.linspace(-1, 1, 80, dtype=ddouble), 150)
    U, s, VT = xprec.linalg.svd(A)

Licence
-------
The xprec library is
Copyright (C) 2021 Markus Wallerberger.
Licensed under the MIT license (see LICENSE.txt).

Contains code from the QD library, which is
Copyright (C) 2012 Yozo Hida, Xiaoye S. Li, David H. Bailey.
Released under the BSD license.
