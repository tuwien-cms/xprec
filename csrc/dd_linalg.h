/* Double-double linear algebra library
 *
 * Implementations were partly inspired by LAPACK, partly from Fredrik
 * Johansson's excellent MPMATH library.
 *
 * Copyright (C) 2021 Markus Wallerberger and others
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "dd_arith.h"

/**
 * Apply Givens rotation to vector:
 *
 *      [ a ]  =  [  c   s ] [ x ]
 *      [ b ]     [ -s   c ] [ y ]
 */
static inline void lmul_givensq(
        ddouble *a, ddouble *b, ddouble c, ddouble s, ddouble x, ddouble y)
{
    *a = addqq(mulqq(c, x), mulqq(s, y));
    *b = subqq(mulqq(c, y), mulqq(s, x));
}

ddouble normq(const ddouble *x, long nn, long sxn);

void rank1updateq(ddouble *a, long ais, long ajs, const ddouble *v, long vs,
                  const ddouble *w, long ws, long ii, long jj);

void givensq(ddouble f, ddouble g, ddouble *c, ddouble *s, ddouble *r);

ddouble householderq(const ddouble *x, ddouble *v, long nn, long sx, long sv);

void svd_tri2x2(ddouble f, ddouble g, ddouble h, ddouble *smin, ddouble *smax,
                ddouble *cv, ddouble *sv, ddouble *cu, ddouble *su);

void golub_kahan_chaseq(ddouble *d, long sd, ddouble *e, long se, long ii,
                        ddouble shift, ddouble *rot);
