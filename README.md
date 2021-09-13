Library for double-double arithmetic calculation
================================================

Extension module for numpy providing the `ddouble` data type.

Loading this module registers an additional scalar data type `ddouble` with
numpy implementing double-double arithmetic.  You can use use the data type
by passing `dtype=xprec.ddouble` to numpy functions.

Installation
------------

    $ pip install xprec

Quickstart
----------

    import numpy as np
    x = np.linspace(0, np.pi)

    # import double-double precision data type
    import xprec as ddouble
    x = x.astype(ddouble)
    y = x * x + 1
    z = np.sin(x)

Licence
-------
The xprec library is
Copyright (C) 2021 Markus Wallerberger
Licensed under the MIT license (see LICENSE.txt).

Contains code from the QD library, which is
Copyright (C) 2012 Yozo Hida, Xiaoye S. Li, David H. Bailey
Released under the BSD license
