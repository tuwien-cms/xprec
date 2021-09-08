Library for double-double arithmetic calculation
================================================

To install from checked out branch:

    $ pip install xprec

Quickstart:

    import numpy as np
    x = np.linspace(0, np.pi)

    # convert to extra precision and do calculations.
    import xprec as xp
    x = xp.ddarray(x)
    y = x * x + 1
    z = np.sin(x)
