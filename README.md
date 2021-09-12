Library for double-double arithmetic calculation
================================================

To install from checked out branch:

    $ pip install xprec

Quickstart:

    import numpy as np
    x = np.linspace(0, np.pi)

    # import double-double precision data type
    import xprec as ddouble
    x = x.astype(ddouble)
    y = x * x + 1
    z = np.sin(x)
