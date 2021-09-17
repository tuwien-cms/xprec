/* Double-double arithmetic library
 *
 * Part of the functions are copied from the QD library for U.C. Berkeley
 * and licensed BSD.
 *
 * Copyright (C) 2012 Yozo Hida, Xiaoye S. Li, David H. Bailey
 * Copyright (C) 2021 Markus Wallerberger and others
 * SPDX-License-Identifier: MIT, BSD
 */
#include "dd_arith.h"

// 2**500 and 2**(-500);
static const double LARGE = 3.273390607896142e+150;
static const double INV_LARGE = 3.054936363499605e-151;

static ddouble hypotqq_compute(ddouble x, ddouble y)
{
    return sqrtq(addqq(sqrq(x), sqrq(y)));
}

ddouble _hypotqq_ordered(ddouble x, ddouble y)
{
    // assume that x >= y >= 0
    // special cases
    if (iszeroq(x))
        return x;

    // if very large or very small, renormalize
    if (x.hi > LARGE) {
        x = mul_pwr2(x, INV_LARGE);
        y = mul_pwr2(y, INV_LARGE);
        return mul_pwr2(hypotqq_compute(x, y), LARGE);
    }
    if (x.hi < INV_LARGE) {
        x = mul_pwr2(x, LARGE);
        y = mul_pwr2(y, LARGE);
        return mul_pwr2(hypotqq_compute(x, y), INV_LARGE);
    }

    // normal case
    return hypotqq_compute(x, y);
}

ddouble sqrtq(ddouble a)
{
    /* Given approximation x to 1/sqrt(a), perform a single Newton step:
     *
     *    sqrt(a) = a*x + [a - (a*x)^2] * x / 2   (approx)
     *
     * The approximation is accurate to twice the accuracy of x.
     * Also, the multiplication (a*x) and [-]*x can be done with
     * only half the precision.
     * From: Karp, High Precision Division and Square Root, 1993
     */
    if (a.hi <= 0)
        return (ddouble){sqrt(a.hi), 0};

    double x = 1.0 / sqrt(a.hi);
    double ax = a.hi * x;
    ddouble ax_sqr = sqrq((ddouble){ax, 0});
    double diff = subqq(a, ax_sqr).hi * x * 0.5;
    return two_sum(ax, diff);
}

/* Inverse Factorials from 1/3!, 1/4!, asf. */
static int _n_inv_fact = 15;
static const ddouble _inv_fact[] = {
    {1.66666666666666657e-01, 9.25185853854297066e-18},
    {4.16666666666666644e-02, 2.31296463463574266e-18},
    {8.33333333333333322e-03, 1.15648231731787138e-19},
    {1.38888888888888894e-03, -5.30054395437357706e-20},
    {1.98412698412698413e-04, 1.72095582934207053e-22},
    {2.48015873015873016e-05, 2.15119478667758816e-23},
    {2.75573192239858925e-06, -1.85839327404647208e-22},
    {2.75573192239858883e-07, 2.37677146222502973e-23},
    {2.50521083854417202e-08, -1.44881407093591197e-24},
    {2.08767569878681002e-09, -1.20734505911325997e-25},
    {1.60590438368216133e-10, 1.25852945887520981e-26},
    {1.14707455977297245e-11, 2.06555127528307454e-28},
    {7.64716373181981641e-13, 7.03872877733453001e-30},
    {4.77947733238738525e-14, 4.39920548583408126e-31},
    {2.81145725434552060e-15, 1.65088427308614326e-31}
    };

/**
 * For the exponential of `a`, return compute tuple `x, m` such that:
 *
 *      exp(a) = ldexp(1 + x, m),
 *
 * where `m` is chosen such that `abs(x) < 1`.  The value `x` is returned,
 * whereas the value `m` is given as an out parameter.
 */
static ddouble _exp_reduced(ddouble a, int *m)
{
    /* Strategy:  We first reduce the size of x by noting that
     *
     *     exp(k * r + m * log(2)) = 2^m * exp(r)^k
     *
     * where m and k are integers.  By choosing m appropriately
     * we can make |k * r| <= log(2) / 2 = 0.347.
     */
    const double k = 512.0;
    const double inv_k = 1.0 / k;
    double mm = floor(a.hi / Q_LOG2.hi + 0.5);
    ddouble r = mul_pwr2(subqq(a, mulqd(Q_LOG2, mm)), inv_k);
    *m = (int)mm;

    /* Now, evaluate exp(r) using the familiar Taylor series.  Reducing the
     * argument substantially speeds up the convergence.  First, we compute
     * terms of order 1 and 2 and add it to the sum
     */
    ddouble sum, term, rpower;
    rpower = sqrq(r);
    sum = addqq(r, mul_pwr2(rpower, 0.5));

    /* Next, compute terms of order 3 and up */
    rpower = mulqq(rpower, r);
    term = mulqq(rpower, _inv_fact[0]);
    int i = 0;
    do {
        sum = addqq(sum, term);
        rpower = mulqq(rpower, r);
        ++i;
        term = mulqq(rpower, _inv_fact[i]);
    } while (fabs(term.hi) > inv_k * Q_EPS.hi && i < 5);
    sum = addqq(sum, term);

    /* We now have that approximately exp(r) == 1 + sum.  Raise that to
     * the m'th (512) power by squaring the binomial nine times
     */
    sum = addqq(mul_pwr2(sum, 2.0), sqrq(sum));
    sum = addqq(mul_pwr2(sum, 2.0), sqrq(sum));
    sum = addqq(mul_pwr2(sum, 2.0), sqrq(sum));
    sum = addqq(mul_pwr2(sum, 2.0), sqrq(sum));
    sum = addqq(mul_pwr2(sum, 2.0), sqrq(sum));
    sum = addqq(mul_pwr2(sum, 2.0), sqrq(sum));
    sum = addqq(mul_pwr2(sum, 2.0), sqrq(sum));
    sum = addqq(mul_pwr2(sum, 2.0), sqrq(sum));
    sum = addqq(mul_pwr2(sum, 2.0), sqrq(sum));
    return sum;
}

ddouble expq(ddouble a)
{
    if (a.hi <= -709.0)
        return Q_ZERO;
    if (a.hi >= 709.0)
        return infq();
    if (iszeroq(a))
        return Q_ONE;
    if (isoneq(a))
        return Q_E;

    int m;
    ddouble sum = _exp_reduced(a, &m);

    /** Add back the one and multiply by 2 to the m */
    sum = addqd(sum, 1.0);
    return ldexpq(sum, (int)m);
}

ddouble expm1q(ddouble a)
{
    if (a.hi <= -709.0)
        return (ddouble){-1.0, 0.0};
    if (a.hi >= 709.0)
        return infq();
    if (iszeroq(a))
        return Q_ZERO;

    int m;
    ddouble sum = _exp_reduced(a, &m);

    /* Truncation case: simply return sum */
    if (m == 0)
        return sum;

    /* Non-truncation case: compute full exp, then remove the one */
    sum = addqd(sum, 1.0);
    sum = ldexpq(sum, (int)m);
    return subqd(sum, 1.0);
}

ddouble logq(ddouble a)
{
    /* Strategy.  The Taylor series for log converges much more
     * slowly than that of exp, due to the lack of the factorial
     * term in the denominator.  Hence this routine instead tries
     * to determine the root of the function
     *
     *     f(x) = exp(x) - a
     *
     * using Newton iteration.  The iteration is given by
     *
     *     x' = x - f(x)/f'(x)
     *        = x - (1 - a * exp(-x))
     *        = x + a * exp(-x) - 1.
     *
     * Only one iteration is needed, since Newton's iteration
     * approximately doubles the number of digits per iteration.
     */
    if (isoneq(a))
        return Q_ZERO;
    if (a.hi <= 0.0)
        return nanq();

    ddouble x = {log(a.hi), 0.0}; /* Initial approximation */
    x = subqd(addqq(x, mulqq(a, expq(negq(x)))), 1.0);
    return x;
}

static const ddouble _pi_16 =
    {1.963495408493620697e-01, 7.654042494670957545e-18};

/* Table of sin(k * pi/16) and cos(k * pi/16). */
static const ddouble _sin_table[] = {
    {1.950903220161282758e-01, -7.991079068461731263e-18},
    {3.826834323650897818e-01, -1.005077269646158761e-17},
    {5.555702330196021776e-01, 4.709410940561676821e-17},
    {7.071067811865475727e-01, -4.833646656726456726e-17}
    };

static const ddouble _cos_table[] = {
    {9.807852804032304306e-01, 1.854693999782500573e-17},
    {9.238795325112867385e-01, 1.764504708433667706e-17},
    {8.314696123025452357e-01, 1.407385698472802389e-18},
    {7.071067811865475727e-01, -4.833646656726456726e-17}
    };

static ddouble sin_taylor(ddouble a)
{
    const double thresh = 0.5 * fabs(a.hi) * Q_EPS.hi;
    ddouble r, s, t, x;

    if (iszeroq(a))
        return Q_ZERO;

    int i = 0;
    x = negq(sqrq(a));
    s = a;
    r = a;
    do {
        r = mulqq(r, x);
        t = mulqq(r, _inv_fact[i]);
        s = addqq(s, t);
        i += 2;
    } while (i < _n_inv_fact && fabs(t.hi) > thresh);

    return s;
}

static ddouble cos_taylor(ddouble a)
{
    const double thresh = 0.5 * Q_EPS.hi;
    ddouble r, s, t, x;

    if (iszeroq(a))
        return Q_ONE;

    x = negq(sqrq(a));
    r = x;
    s = adddq(1.0, mul_pwr2(r, 0.5));
    int i = 1;
    do {
        r = mulqq(r, x);
        t = mulqq(r, _inv_fact[i]);
        s = addqq(s, t);
        i += 2;
    } while (i < _n_inv_fact && fabs(t.hi) > thresh);

    return s;
}

static void sincos_taylor(ddouble a, ddouble *sin_a, ddouble *cos_a)
{
    if (iszeroq(a)) {
        *sin_a = Q_ZERO;
        *cos_a = Q_ONE;
    } else {
        *sin_a = sin_taylor(a);
        *cos_a = sqrtq(subdq(1.0, sqrq(*sin_a)));
    }
}

ddouble sinq(ddouble a)
{
    /* Strategy.  To compute sin(x), we choose integers a, b so that
     *
     *   x = s + a * (pi/2) + b * (pi/16)
     *
     * and |s| <= pi/32.  Using the fact that
     *
     *   sin(pi/16) = 0.5 * sqrt(2 - sqrt(2 + sqrt(2)))
     *
     * we can compute sin(x) from sin(s), cos(s).  This greatly
     * increases the convergence of the sine Taylor series.
     */
    if (iszeroq(a))
        return Q_ZERO;

    // approximately reduce modulo 2*pi
    ddouble z = roundq(divqq(a, Q_2PI));
    ddouble r = subqq(a, mulqq(Q_2PI, z));

    // approximately reduce modulo pi/2 and then modulo pi/16.
    ddouble t;
    double q = floor(r.hi / Q_PI_2.hi + 0.5);
    t = subqq(r, mulqd(Q_PI_2, q));
    int j = (int)q;
    q = floor(t.hi / _pi_16.hi + 0.5);
    t = subqq(t, mulqd(_pi_16, q));
    int k = (int)q;
    int abs_k = abs(k);

    if (j < -2 || j > 2)
        return nanq();

    if (abs_k > 4)
        return nanq();

    if (k == 0) {
        switch (j)
        {
        case 0:
            return sin_taylor(t);
        case 1:
            return cos_taylor(t);
        case -1:
            return negq(cos_taylor(t));
        default:
            return negq(sin_taylor(t));
        }
    }

    ddouble u = _cos_table[abs_k - 1];
    ddouble v = _sin_table[abs_k - 1];
    ddouble sin_x, cos_x;
    sincos_taylor(t, &sin_x, &cos_x);
    if (j == 0) {
        if (k > 0)
            r = addqq(mulqq(u, sin_x), mulqq(v, cos_x));
        else
            r = subqq(mulqq(u, sin_x), mulqq(v, cos_x));
    } else if (j == 1) {
        if (k > 0)
            r = subqq(mulqq(u, cos_x), mulqq(v, sin_x));
        else
            r = addqq(mulqq(u, cos_x), mulqq(v, sin_x));
    } else if (j == -1) {
        if (k > 0)
            r = subqq(mulqq(v, sin_x), mulqq(u, cos_x));
        else if (k < 0)   /* NOTE! */
            r = subqq(mulqq(negq(u), cos_x), mulqq(v, sin_x));
    } else {
        if (k > 0)
            r = subqq(mulqq(negq(u), sin_x), mulqq(v, cos_x));
        else
            r = subqq(mulqq(v, cos_x), mulqq(u, sin_x));
    }
    return r;
}

ddouble cosq(ddouble a)
{
    if (iszeroq(a))
        return Q_ONE;

    // approximately reduce modulo 2*pi
    ddouble z = roundq(divqq(a, Q_2PI));
    ddouble r = subqq(a, mulqq(Q_2PI, z));

    // approximately reduce modulo pi/2 and then modulo pi/16.
    ddouble t;
    double q = floor(r.hi / Q_PI_2.hi + 0.5);
    t = subqq(r, mulqd(Q_PI_2, q));
    int j = (int)q;
    q = floor(t.hi / _pi_16.hi + 0.5);
    t = subqq(t, mulqd(_pi_16, q));
    int k = (int)q;
    int abs_k = abs(k);

    if (j < -2 || j > 2)
        return nanq();

    if (abs_k > 4)
        return nanq();

    if (k == 0) {
        switch (j) {
        case 0:
            return cos_taylor(t);
        case 1:
            return negq(sin_taylor(t));
        case -1:
            return sin_taylor(t);
        default:
            return negq(cos_taylor(t));
        }
    }

    ddouble sin_x, cos_x;
    sincos_taylor(t, &sin_x, &cos_x);
    ddouble u = _cos_table[abs_k - 1];
    ddouble v = _sin_table[abs_k - 1];

    if (j == 0) {
        if (k > 0)
            r = subqq(mulqq(u, cos_x), mulqq(v, sin_x));
        else
            r = addqq(mulqq(u, cos_x), mulqq(v, sin_x));
    } else if (j == 1) {
        if (k > 0)
            r = subqq(mulqq(negq(u), sin_x), mulqq(v, cos_x));
        else
            r = subqq(mulqq(v, cos_x), mulqq(u, sin_x));
    } else if (j == -1) {
        if (k > 0)
            r = addqq(mulqq(u, sin_x), mulqq(v, cos_x));
        else
            r = subqq(mulqq(u, sin_x), mulqq(v, cos_x));
    } else {
        if (k > 0)
            r = subqq(mulqq(v, sin_x), mulqq(u, cos_x));
        else
            r = subqq(mulqq(negq(u), cos_x), mulqq(v, sin_x));
    }
    return r;
}

ddouble sinhq(ddouble a)
{
    if (iszeroq(a))
        return Q_ZERO;

    if (absq(a).hi > 0.05) {
        ddouble ea = expq(a);
        if (isinfq(ea))
            return ea;
        if (iszeroq(ea))
            return negq(infq());
        return mul_pwr2(subqq(ea, reciprocalq(ea)), 0.5);
    }

    /* since a is small, using the above formula gives
     * a lot of cancellation.  So use Taylor series.
     */
    ddouble s = a;
    ddouble t = a;
    ddouble r = sqrq(t);
    double m = 1.0;
    double thresh = fabs((a.hi) * Q_EPS.hi);

    do {
        m += 2.0;
        t = mulqq(t, r);
        t = divqd(t, (m - 1) * m);
        s = addqq(s, t);
    } while (absq(t).hi > thresh);
    return s;
}

ddouble coshq(ddouble a)
{
    if (iszeroq(a))
        return Q_ONE;

    ddouble ea = expq(a);
    if (isinfq(ea) || iszeroq(ea))
        return infq();
    return mul_pwr2(addqq(ea, reciprocalq(ea)), 0.5);
}

ddouble tanhq(ddouble a)
{
    if (iszeroq(a))
        return Q_ZERO;

    if (fabs(a.hi) > 0.05) {
        ddouble ea = expq(a);
        ddouble inv_ea = reciprocalq(ea);
        return divqq(subqq(ea, inv_ea), addqq(ea, inv_ea));
    }

    ddouble s, c;
    s = sinhq(a);
    c = sqrtq(adddq(1.0, sqrq(s)));
    return divqq(s, c);
}
