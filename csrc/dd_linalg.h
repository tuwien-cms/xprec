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
    *a = addww(mulww(c, x), mulww(s, y));
    *b = subww(mulww(c, y), mulww(s, x));
}

/** Compute 2-norm of a vector */
ddouble normw(const ddouble *x, long nn, long sxn);

/**
 * Perform a rank-one update of a `ii` times `jj` matrix:
 *
 *       A[i, j] += v[i] * w[j]
 */
void rank1updateq(ddouble *a, long ais, long ajs, const ddouble *v, long vs,
                  const ddouble *w, long ws, long ii, long jj);

/**
 * Compute Givens rotation `R` matrix that satisfies:
 *
 *      [  c  s ] [ f ]     [ r ]
 *      [ -s  c ] [ g ]  =  [ 0 ]
 */
void givensw(ddouble f, ddouble g, ddouble *c, ddouble *s, ddouble *r);

/**
 * Compute Householder reflector `H[tau, v]`, defined as:
 *
 *      H[tau, v] = I - tau * v @ v.T
 *
 * that, when applied to a given `x`, zeros out all but the first component.
 * The scaling factor `tau` is returned, while `v` is written.
 */
ddouble householderw(const ddouble *x, ddouble *v, long nn, long sx, long sv);

/**
 * Perform the SVD of an arbitrary two-by-two matrix:
 *
 *      [ a11  a12 ]  =  [  cu  -su ] [ smax     0 ] [  cv   sv ]
 *      [ a21  a22 ]     [  su   cu ] [    0  smin ] [ -sv   cv ]
 */
void svd_2x2(ddouble a11, ddouble a12, ddouble a21, ddouble a22, ddouble *smin,
             ddouble *smax, ddouble *cv, ddouble *sv, ddouble *cu, ddouble *su);



ddouble jacobi_sweep(ddouble *u, long sui, long suj, ddouble *vt, long svi,
                     long svj, long ii, long jj);


void golub_kahan_chaseq(ddouble *d, long sd, ddouble *e, long se, long ii,
                        ddouble *rot);
