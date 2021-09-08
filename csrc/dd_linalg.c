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
        sum = addqq(sum, sqrq(curr));
    };
    return mul_pwr2(sqrtq(sum), 1.0/scaling);
}

ddouble normq(const ddouble *x, long nn, long sxn)
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

void givensq(ddouble f, ddouble g, ddouble *c, ddouble *s, ddouble *r)
{
    /* ACM Trans. Math. Softw. 28(2), 206, Alg 1 */
    if (iszeroq(g)) {
        *c = Q_ONE;
        *s = Q_ZERO;
        *r = f;
    } else if (iszeroq(f)) {
        *c = Q_ZERO;
        *s = (ddouble) {signbitq(g), 0.0};
        *r = absq(g);
    } else {
        *r = copysignqq(hypotqq(f, g), f);

        /* This may come at a slight loss of precision, however, we should
         * not really have to care ...
         */
        ddouble inv_r = reciprocalq(*r);
        *c = mulqq(f, inv_r);
        *s = mulqq(g, inv_r);
    }
}

void svd_tri2x2(ddouble f, ddouble g, ddouble h, ddouble *smin, ddouble *smax,
                ddouble *cv, ddouble *sv, ddouble *cu, ddouble *su)
{
    ddouble fa = absq(f);
    ddouble ga = absq(g);
    ddouble ha = absq(h);
    bool compute_uv = cv != NULL;

    if (lessqq(fa, ha)) {
        // switch h <-> f, cu <-> sv, cv <-> su
        svd_tri2x2(h, g, f, smin, smax, su, cu, sv, cv);
        return;
    }
    if (iszeroq(ga)) {
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
            *smin = divqq(fa, divqq(ga, ha));
        else
            *smin = mulqq(divqq(fa, ga), ha);
        if (compute_uv) {
            *cu = Q_ONE;
            *su = divqq(h, g);
            *cv = Q_ONE;
            *sv = divqq(f, g);
        }
        return;
    }
    // normal case
    ddouble fmh = subqq(fa, ha);
    ddouble d = divqq(fmh, fa);
    ddouble q = divqq(g, f);
    ddouble s = subdq(2.0, d);
    ddouble spq = hypotqq(q, s);
    ddouble dpq = hypotqq(d, q);
    ddouble a = mul_pwr2(addqq(spq, dpq), 0.5);
    *smin = absq(divqq(ha, a));
    *smax = absq(mulqq(fa, a));

    if (compute_uv) {
        ddouble tmp = addqq(divqq(q, addqq(spq, s)),
                            divqq(q, addqq(dpq, d)));
        tmp = mulqq(tmp, adddq(1.0, a));
        ddouble tt = hypotqd(tmp, 2.0);
        *cv = divdq(2.0, tt);
        *sv = divqq(tmp, tt);
        *cu = divqq(addqq(*cv, mulqq(*sv, q)), a);
        *su = divqq(mulqq(divqq(h, f), *sv), a);
    }
}
