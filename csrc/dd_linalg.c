/* Double-double linear algebra library
 *
 * Implementations were partly inspired by LAPACK, partly from Fredrik
 * Johansson's excellent MPMATH library.
 *
 * Copyright (C) 2021 Markus Wallerberger and others
 * SPDX-License-Identifier: MIT
 */
#include "dd_linalg.h"

// 2**500 and 2**(-500);
static const double LARGE = 3.273390607896142e+150;
static const double INV_LARGE = 3.054936363499605e-151;

static ddouble normq_scaled(const ddouble *x, long nn, long sxn,
                            double scaling)
{
    ddouble sum = Q_ZERO;
    for (long n = 0; n < nn; ++n, x += sxn) {
        ddouble curr = mul_pwr2(*x, scaling);
        sum = addww(sum, sqrw(curr));
    };
    return mul_pwr2(sqrtw(sum), 1.0/scaling);
}

ddouble normw(const ddouble *x, long nn, long sxn)
{
    ddouble sum = normq_scaled(x, nn, sxn, 1.0);

    // fall back to other routines in case of over/underflow
    if (sum.hi > LARGE)
        return normq_scaled(x, nn, sxn, INV_LARGE);
    else if (sum.hi < INV_LARGE)
        return normq_scaled(x, nn, sxn, LARGE);
    else
        return sum;
}

ddouble householderw(const ddouble *x, ddouble *v, long nn, long sx, long sv)
{
    if (nn == 0)
        return Q_ZERO;

    ddouble norm_x = normw(x + sx, nn - 1, sx);
    if (iszerow(norm_x))
        return Q_ZERO;

    ddouble alpha = *x;
    ddouble beta = copysignww(hypotww(alpha, norm_x), alpha);

    ddouble diff = subww(beta, alpha);
    ddouble tau = divww(diff, beta);
    ddouble scale = reciprocalw(negw(diff));

    v[0] = Q_ONE;
    for (long n = 1; n != nn; ++n)
        v[n * sv] = mulww(scale, x[n * sx]);
    return tau;
}

void rank1updateq(ddouble *a, long ais, long ajs, const ddouble *v, long vs,
                  const ddouble *w, long ws, long ii, long jj)
{
    #pragma omp parallel for collapse(2)
    for (long i = 0; i < ii; ++i) {
        for (long j = 0; j < jj; ++j) {
            ddouble tmp = mulww(v[i * vs], w[j * ws]);
            a[i * ais + j * ajs] = addww(a[i * ais + j * ajs], tmp);
        }
    }
}

void givensw(ddouble f, ddouble g, ddouble *c, ddouble *s, ddouble *r)
{
    /* ACM Trans. Math. Softw. 28(2), 206, Alg 1 */
    if (iszerow(g)) {
        *c = Q_ONE;
        *s = Q_ZERO;
        *r = f;
    } else if (iszerow(f)) {
        *c = Q_ZERO;
        *s = (ddouble) {signbitw(g), 0.0};
        *r = absw(g);
    } else {
        *r = copysignww(hypotww(f, g), f);

        /* This may come at a slight loss of precision, however, we should
         * not really have to care ...
         */
        ddouble inv_r = reciprocalw(*r);
        *c = mulww(f, inv_r);
        *s = mulww(g, inv_r);
    }
}

static void svd_tri2x2(
                ddouble f, ddouble g, ddouble h, ddouble *smin, ddouble *smax,
                ddouble *cv, ddouble *sv, ddouble *cu, ddouble *su)
{
    ddouble fa = absw(f);
    ddouble ga = absw(g);
    ddouble ha = absw(h);
    bool compute_uv = cv != NULL;

    if (lessww(fa, ha)) {
        // switch h <-> f, cu <-> sv, cv <-> su
        svd_tri2x2(h, g, f, smin, smax, su, cu, sv, cv);
        return;
    }
    if (iszerow(ga)) {
        // already diagonal
        *smin = ha;
        *smax = fa;
        if (compute_uv) {
            *cu = Q_ONE;
            *su = Q_ZERO;
            *cv = Q_ONE;
            *sv = Q_ZERO;
        }
        return;
    }
    if (fa.hi < Q_EPS.hi * ga.hi) {
        // ga is very large
        *smax = ga;
        if (ha.hi > 1.0)
            *smin = divww(fa, divww(ga, ha));
        else
            *smin = mulww(divww(fa, ga), ha);
        if (compute_uv) {
            *cu = Q_ONE;
            *su = divww(h, g);
            *cv = Q_ONE;
            *sv = divww(f, g);
        }
        return;
    }
    // normal case
    ddouble fmh = subww(fa, ha);
    ddouble d = divww(fmh, fa);
    ddouble q = divww(g, f);
    ddouble s = subdw(2.0, d);
    ddouble spw = hypotww(q, s);
    ddouble dpw = hypotww(d, q);
    ddouble a = mul_pwr2(addww(spw, dpw), 0.5);
    *smin = absw(divww(ha, a));
    *smax = absw(mulww(fa, a));

    if (compute_uv) {
        ddouble tmp = addww(divww(q, addww(spw, s)),
                            divww(q, addww(dpw, d)));
        tmp = mulww(tmp, adddw(1.0, a));
        ddouble tt = hypotwd(tmp, 2.0);
        *cv = divdw(2.0, tt);
        *sv = divww(tmp, tt);
        *cu = divww(addww(*cv, mulww(*sv, q)), a);
        *su = divww(mulww(divww(h, f), *sv), a);
    }
}

void svd_2x2(ddouble a11, ddouble a12, ddouble a21, ddouble a22, ddouble *smin,
             ddouble *smax, ddouble *cv, ddouble *sv, ddouble *cu, ddouble *su)
{
    bool compute_uv = cv != NULL;
    if(iszerow(a21))
        return svd_tri2x2(a11, a12, a22, smin, smax, cv, sv, cu, su);

    /* First, we use a givens rotation  Rx
     *   [  cx   sx ] [ a11  a12 ] = [ rx  a12' ]
     *   [ -sx   cx ] [ a21  a22 ]   [ 0   a22' ]
     */
    ddouble cx, sx, rx;
    givensw(a11, a21, &cx, &sx, &rx);
    a11 = rx;
    a21 = Q_ZERO;
    lmul_givensq(&a12, &a22, cx, sx, a12, a22);

    /* Next, use the triangular routine
     *    [ f  g ]  =  [  cu  -su ] [ smax     0 ] [  cv   sv ]
     *    [ 0  h ]     [  su   cu ] [    0  smin ] [ -sv   cv ]
     */
    svd_tri2x2(a11, a12, a22, smin, smax, cv, sv, cu, su);

    /* Finally, update the LHS (U) transform as follows:
     *   [  cx  -sx ] [  cu  -su ] = [  cu'  -su' ]
     *   [  sx   cx ] [  su   cu ]   [  su'   cu' ]
     */
    if (compute_uv)
        lmul_givensq(cu, su, cx, negw(sx), *cu, *su);
}

ddouble jacobi_sweep(ddouble *u, long sui, long suj, ddouble *vt, long svi,
                     long svj, long ii, long jj)
{
    ddouble _cu, _su, cv, sv, _smin, _smax;
    ddouble offd = Q_ZERO;

    if (ii < jj)
        return nanw();

    // Note that the inner loop only runs over the square portion!
    for (long i = 0; i < jj - 1; ++i) {
        for (long j = i + 1; j < jj; ++j) {
            // Construct the matrix to be diagonalized
            ddouble Hii = Q_ZERO, Hij = Q_ZERO, Hjj = Q_ZERO;
            for (long k = 0; k != ii; ++k) {
                ddouble u_ki = u[k * sui + i * suj];
                ddouble u_kj = u[k * sui + j * suj];
                Hii = addww(Hii, mulww(u_ki, u_ki));
                Hij = addww(Hij, mulww(u_ki, u_kj));
                Hjj = addww(Hjj, mulww(u_kj, u_kj));
            }
            offd = addww(offd, sqrw(Hij));

            // diagonalize
            svd_2x2(Hii, Hij, Hij, Hjj, &_smin, &_smax, &cv, &sv, &_cu, &_su);

            // apply rotation to VT
            for (long k = 0; k < jj; ++k) {
                ddouble *vt_ik = &vt[i * svi + k * svj];
                ddouble *vt_jk = &vt[j * svi + k * svj];
                lmul_givensq(vt_ik, vt_jk, cv, sv, *vt_ik, *vt_jk);
            }

            // apply transposed rotation to U
            for (long k = 0; k < ii; ++k) {
                ddouble *u_ki = &u[k * sui + i * suj];
                ddouble *u_kj = &u[k * sui + j * suj];
                lmul_givensq(u_ki, u_kj, cv, sv, *u_ki, *u_kj);
            }
        }
    }
    offd = sqrtw(offd);
    return offd;
}

static ddouble gk_shift(ddouble d1, ddouble e1, ddouble d2)
{
    /* Get singular values of 2x2 triangular matrix formed from the lower
     * right corner in the array:
     *
     *      [ d[ii-2]  e[ii-2] ]
     *      [ 0        d[ii-1] ]
     */
    ddouble smin, smax;
    svd_tri2x2(d1, e1, d2, &smin, &smax, NULL, NULL, NULL, NULL);

    ddouble smin_dist = absw(subww(smin, d2));
    ddouble smax_dist = absw(subww(smax, d2));
    return lessww(smin_dist, smax_dist) ? smin : smax;
}

void golub_kahan_chaseq(ddouble *d, long sd, ddouble *e, long se, long ii,
                        ddouble *rot)
{
    if (ii < 2)
        return;

    ddouble shift = gk_shift(d[(ii-2)*sd], e[(ii-2)*se], d[(ii-1)*sd]);
    ddouble g = e[0];
    ddouble f = addww(copysigndw(1.0, d[0]), divww(shift, d[0]));
    f = mulww(f, subww(absw(d[0]), shift));

    for (long i = 0; i < (ii - 1); ++i) {
        ddouble r, cosr, sinr;
        givensw(f, g, &cosr, &sinr, &r);
        if (i != 0)
            e[(i-1)*se] = r;

        lmul_givensq(&f, &e[i*se], cosr, sinr, d[i*sd], e[i*se]);
        lmul_givensq(&g, &d[(i+1)*sd], cosr, sinr, Q_ZERO, d[(i+1)*sd]);
        *(rot++) = cosr;
        *(rot++) = sinr;

        ddouble cosl, sinl;
        givensw(f, g, &cosl, &sinl, &r);
        d[i*sd] = r;
        lmul_givensq(&f, &d[(i+1)*sd], cosl, sinl, e[i*se], d[(i+1)*sd]);
        if (i < ii - 2) {
            lmul_givensq(&g, &e[(i+1)*se], cosl, sinl, Q_ZERO, e[(i+1)*se]);
        }
        *(rot++) = cosl;
        *(rot++) = sinl;
    }
    e[(ii-2)*se] = f;
}
