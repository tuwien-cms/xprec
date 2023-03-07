/* Double-double arithmetic library
 *
 * Part of the functions are modified from the QD library for U.C. Berkeley
 * and licensed under a modified BSD license (see QD-LICENSE.txt)
 *
 * Some of the algorithms were updated according to the findings in
 * M. Joldes, et al., ACM Trans. Math. Softw. 44, 1-27 (2018)
 * (Algorithm numbers in the code)
 *
 * Copyright (C) 2012 Yozo Hida, Xiaoye S. Li, David H. Bailey
 * Copyright (C) 2021 Markus Wallerberger and others
 * SPDX-License-Identifier: MIT and Modified-BSD
 */
#pragma once
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

/**
 * Type for double-double calculations
 */
typedef struct {
    double hi;
    double lo;
} ddouble;

static inline ddouble two_sum_quick(double a, double b)
{
    double s = a + b;
    double lo = b - (s - a);
    return (ddouble){.hi = s, .lo = lo};
}

static inline ddouble two_sum(double a, double b)
{
    double s = a + b;
    double v = s - a;
    double lo = (a - (s - v)) + (b - v);
    return (ddouble){.hi = s, .lo = lo};
}

static inline ddouble two_diff(double a, double b)
{
    double s = a - b;
    double v = s - a;
    double lo = (a - (s - v)) - (b + v);
    return (ddouble){.hi = s, .lo = lo};
}

static inline ddouble two_prod(double a, double b)
{
    double s = a * b;
    double lo = fma(a, b, -s);
    return (ddouble){.hi = s, .lo = lo};
}

/* -------------------- Combining quad/double ------------------------ */

static inline ddouble addqd(ddouble x, double y)
{
    ddouble s = two_sum(x.hi, y);
    double v = x.lo + s.lo;
    return two_sum_quick(s.hi, v);
}

static inline ddouble subqd(ddouble x, double y)
{
    ddouble s = two_diff(x.hi, y);
    double v = x.lo + s.lo;
    return two_sum_quick(s.hi, v);
}

static inline ddouble mulqd(ddouble x, double y)
{
    ddouble c = two_prod(x.hi, y);
    double v = fma(x.lo, y, c.lo);
    return two_sum_quick(c.hi, v);
}

static inline ddouble divqd(ddouble x, double y)
{
    /* Alg 14 */
    double t_hi = x.hi / y;
    ddouble pi = two_prod(t_hi, y);
    double d_hi = x.hi - pi.hi;
    double d_lo = x.lo - pi.lo;
    double t_lo = (d_hi + d_lo) / y;
    return two_sum_quick(t_hi, t_lo);
}

/* -------------------- Combining double/quad ------------------------- */

static inline ddouble negw(ddouble);
static inline ddouble reciprocalw(ddouble);

static inline ddouble adddw(double x, ddouble y)
{
    return addqd(y, x);
}

static inline ddouble subdw(double x, ddouble y)
{
    /* TODO: Probably not ideal */
    return addqd(negw(y), x);
}

static inline ddouble muldw(double x, ddouble y)
{
    return mulqd(y, x);
}

static inline ddouble divdw(double x, ddouble y)
{
    /* TODO: Probably not ideal */
    return mulqd(reciprocalw(y), x);
}

static inline ddouble mul_pwr2(ddouble a, double b) {
    return (ddouble){a.hi * b, a.lo * b};
}

/* -------------------- Combining quad/quad ------------------------- */

static inline ddouble addqq(ddouble x, ddouble y)
{
    ddouble s = two_sum(x.hi, y.hi);
    ddouble t = two_sum(x.lo, y.lo);
    ddouble v = two_sum_quick(s.hi, s.lo + t.hi);
    ddouble z = two_sum_quick(v.hi, t.lo + v.lo);
    return z;
}

static inline ddouble subqq(ddouble x, ddouble y)
{
    ddouble s = two_diff(x.hi, y.hi);
    ddouble t = two_diff(x.lo, y.lo);
    ddouble v = two_sum_quick(s.hi, s.lo + t.hi);
    ddouble z = two_sum_quick(v.hi, t.lo + v.lo);
    return z;
}

static inline ddouble mulqq(ddouble a, ddouble b)
{
    /* Alg 11 */
    ddouble c = two_prod(a.hi, b.hi);
    double t = a.hi * b.lo;
    t = fma(a.lo, b.hi, t);
    return two_sum_quick(c.hi, c.lo + t);
}

static inline ddouble divqq(ddouble x, ddouble y)
{
    /* Alg 17 */
    double t_hi = x.hi / y.hi;
    ddouble r = mulqd(y, t_hi);
    double pi_hi = x.hi - r.hi;
    double d = pi_hi + (x.lo - r.lo);
    double t_lo = d / y.hi;
    return two_sum_quick(t_hi, t_lo);
}

/* -------------------- Unary functions ------------------------- */

static inline ddouble negw(ddouble a)
{
    return (ddouble){-a.hi, -a.lo};
}

static inline ddouble posw(ddouble a)
{
    return (ddouble){-a.hi, -a.lo};
}

static inline ddouble absw(ddouble a)
{
    return signbit(a.hi) ? negw(a) : a;
}

static inline ddouble reciprocalw(ddouble y)
{
    /* Alg 17 with x = 1 */
    double t_hi = 1.0 / y.hi;
    ddouble r = mulqd(y, t_hi);
    double pi_hi = 1.0 - r.hi;
    double d = pi_hi - r.lo;
    double t_lo = d / y.hi;
    return two_sum_quick(t_hi, t_lo);
}

static inline ddouble sqrw(ddouble a)
{
    /* Alg 11 */
    ddouble c = two_prod(a.hi, a.hi);
    double t = 2 * a.hi * a.lo;
    return two_sum_quick(c.hi, c.lo + t);
}

static inline ddouble roundw(ddouble a)
{
    double hi = round(a.hi);
    double lo;

    if (hi == a.hi) {
        /* High word is an integer already.  Round the low word.*/
        lo = round(a.lo);

        /* Renormalize. This is needed if x[0] = some integer, x[1] = 1/2.*/
        return two_sum_quick(hi, lo);
    } else {
        /* High word is not an integer. */
        lo = 0.0;
        if (fabs(hi - a.hi) == 0.5 && a.lo < 0.0) {
            /* There is a tie in the high word, consult the low word
             * to break the tie.
             * NOTE: This does not cause INEXACT.
             */
            hi -= 1.0;
        }
        return (ddouble){hi, lo};
    }
}

static inline ddouble floorw(ddouble a)
{
    double hi = floor(a.hi);
    double lo = 0.0;

    if (hi == a.hi) {
        /* High word is integer already.  Round the low word. */
        lo = floor(a.lo);
        return two_sum_quick(hi, lo);
    }
    return (ddouble){hi, lo};
}

static inline ddouble ceilw(ddouble a)
{
    double hi = ceil(a.hi);
    double lo = 0.0;

    if (hi == a.hi) {
        /* High word is integer already.  Round the low word. */
        lo = ceil(a.lo);
        return two_sum_quick(hi, lo);
    }
    return (ddouble){hi, lo};
}

static inline bool signbitw(ddouble x)
{
    return signbit(x.hi);
}

static inline ddouble copysignqq(ddouble x, ddouble y)
{
    /* The sign is determined by the hi part, however, the sign of hi and lo
     * need not be the same, so we cannot merely broadcast copysign to both
     * parts.
     */
    return signbitw(x) != signbitw(y) ? negw(x) : x;
}

static inline ddouble copysignqd(ddouble x, double y)
{
    return signbitw(x) != signbit(y) ? negw(x) : x;
}

static inline ddouble copysigndw(double x, ddouble y)
{
    /* It is less surprising to return a ddouble here */
    double res = copysign(x, y.hi);
    return (ddouble) {res, 0.0};
}

static inline bool iszerow(ddouble x);

static inline ddouble signw(ddouble x)
{
    /* The numpy sign function does not respect signed zeros.  We do. */
    if (iszerow(x))
        return x;
    return copysigndw(1.0, x);
}

/******************************** Constants *********************************/

static inline ddouble nanw()
{
    double nan = strtod("NaN", NULL);
    return (ddouble){nan, nan};
}

static inline ddouble infw()
{
    double inf = strtod("Inf", NULL);
    return (ddouble){inf, inf};
}

static const ddouble Q_ZERO = {0.0, 0.0};
static const ddouble Q_ONE = {1.0, 0.0};
static const ddouble Q_2PI = {6.283185307179586232e+00, 2.449293598294706414e-16};
static const ddouble Q_PI = {3.141592653589793116e+00, 1.224646799147353207e-16};
static const ddouble Q_PI_2 = {1.570796326794896558e+00, 6.123233995736766036e-17};
static const ddouble Q_PI_4 = {7.853981633974482790e-01, 3.061616997868383018e-17};
static const ddouble Q_3PI_4 = {2.356194490192344837e+00, 9.1848509936051484375e-17};
static const ddouble Q_PI_16 = {1.963495408493620697e-01, 7.654042494670957545e-18};
static const ddouble Q_E = {2.718281828459045091e+00, 1.445646891729250158e-16};
static const ddouble Q_LOG2 = {6.931471805599452862e-01, 2.319046813846299558e-17};
static const ddouble Q_LOG10 = {2.302585092994045901e+00, -2.170756223382249351e-16};

static const ddouble Q_EPS = {4.93038065763132e-32, 0.0};
static const ddouble Q_MIN = {2.0041683600089728e-292, 0.0};
static const ddouble Q_MAX = {1.79769313486231570815e+308, 0.0};
static const ddouble Q_TINY = {2.2250738585072014e-308, 0.0};


static inline bool isfinitew(ddouble x)
{
    return isfinite(x.hi);
}

static inline bool isinfw(ddouble x)
{
    return isinf(x.hi);
}

static inline bool isnanw(ddouble x)
{
    return isnan(x.hi);
}

/*********************** Comparisons q/q ***************************/

static inline bool equalqq(ddouble a, ddouble b)
{
    return a.hi == b.hi && a.lo == b.lo;
}

static inline bool notequalqq(ddouble a, ddouble b)
{
    return a.hi != b.hi || a.lo != b.lo;
}

static inline bool greaterqq(ddouble a, ddouble b)
{
    return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo);
}

static inline bool lessqq(ddouble a, ddouble b)
{
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}

static inline bool greaterequalqq(ddouble a, ddouble b)
{
    return a.hi > b.hi || (a.hi == b.hi && a.lo >= b.lo);
}

static inline bool lessequalqq(ddouble a, ddouble b)
{
    return a.hi < b.hi || (a.hi == b.hi && a.lo <= b.lo);
}

/*********************** Comparisons q/d ***************************/

static inline bool equalqd(ddouble a, double b)
{
    return equalqq(a, (ddouble){b, 0});
}

static inline bool notequalqd(ddouble a, double b)
{
    return notequalqq(a, (ddouble){b, 0});
}

static inline bool greaterqd(ddouble a, double b)
{
    return greaterqq(a, (ddouble){b, 0});
}

static inline bool lessqd(ddouble a, double b)
{
    return lessqq(a, (ddouble){b, 0});
}

static inline bool greaterequalqd(ddouble a, double b)
{
    return greaterequalqq(a, (ddouble){b, 0});
}

static inline bool lessequalqd(ddouble a, double b)
{
    return lessequalqq(a, (ddouble){b, 0});
}

/*********************** Comparisons d/q ***************************/

static inline bool equaldw(double a, ddouble b)
{
    return equalqq((ddouble){a, 0}, b);
}

static inline bool notequaldw(double a, ddouble b)
{
    return notequalqq((ddouble){a, 0}, b);
}

static inline bool greaterdw(double a, ddouble b)
{
    return greaterqq((ddouble){a, 0}, b);
}

static inline bool lessdw(double a, ddouble b)
{
    return lessqq((ddouble){a, 0}, b);
}

static inline bool greaterequaldw(double a, ddouble b)
{
    return greaterequalqq((ddouble){a, 0}, b);
}

static inline bool lessequaldw(double a, ddouble b)
{
    return lessequalqq((ddouble){a, 0}, b);
}

/************************ Minimum/maximum ************************/

static inline ddouble fminqq(ddouble a, ddouble b)
{
    return lessqq(a, b) ? a : b;
}

static inline ddouble fmaxqq(ddouble a, ddouble b)
{
    return greaterqq(a, b) ? a : b;
}

static inline ddouble fminqd(ddouble a, double b)
{
    return lessqd(a, b) ? a : (ddouble) {b, 0};
}

static inline ddouble fmaxqd(ddouble a, double b)
{
    return greaterqd(a, b) ? a : (ddouble) {b, 0};
}

static inline ddouble fmindw(double a, ddouble b)
{
    return lessdw(a, b) ? (ddouble) {a, 0} : b;
}

static inline ddouble fmaxdw(double a, ddouble b)
{
    return greaterdw(a, b) ? (ddouble) {a, 0} : b;
}

/************************** Unary tests **************************/

static inline bool iszerow(ddouble x)
{
    return x.hi == 0.0;
}

static inline bool isonew(ddouble x)
{
    return x.hi == 1.0 && x.lo == 0.0;
}

static inline bool ispositivew(ddouble x)
{
    return x.hi > 0.0;
}

static inline bool isnegativew(ddouble x)
{
    return x.hi < 0.0;
}

/************************** Advanced math functions ********************/

ddouble sqrtw(ddouble a);

static inline ddouble ldexpw(ddouble a, int exp)
{
    return (ddouble) {ldexp(a.hi, exp), ldexp(a.lo, exp)};
}

/************************* Binary functions ************************/

ddouble _hypotqq_ordered(ddouble x, ddouble y);

static inline ddouble hypotqq(ddouble x, ddouble y)
{
    x = absw(x);
    y = absw(y);
    if (x.hi < y.hi)
        return _hypotqq_ordered(y, x);
    else
        return _hypotqq_ordered(x, y);
}

static inline ddouble hypotdw(double x, ddouble y)
{
    return hypotqq((ddouble){x, 0}, y);
}

static inline ddouble hypotqd(ddouble x, double y)
{
    return hypotqq(x, (ddouble){y, 0});
}

/* Computes the nearest integer to d. */
static inline ddouble nintw(ddouble d) {
    if (equalqq(d, floorw(d))) {
        return d;
    }
    return floorw(addqq(d, (ddouble){0.5, 0}));
}

ddouble expw(ddouble a);
ddouble expm1q(ddouble a);
ddouble ldexpqi(ddouble a, int m);
ddouble logw(ddouble a);
ddouble sinw(ddouble a);
ddouble cosw(ddouble a);
ddouble tanw(ddouble a);
ddouble sinhw(ddouble a);
ddouble coshw(ddouble a);
ddouble tanhw(ddouble a);
ddouble atanw(ddouble a);
ddouble acosw(ddouble a);
ddouble asinw(ddouble a);
ddouble atanhw(ddouble a);
ddouble acoshw(ddouble a);
ddouble asinhw(ddouble a);
ddouble atan2qd(ddouble a, double b);
ddouble atan2dq(double a, ddouble b);
ddouble atan2qq(ddouble a, ddouble b);
ddouble powqq(ddouble a, ddouble b);
ddouble powqd(ddouble a, double b);
ddouble powdw(double a, ddouble b);
ddouble modfqq(ddouble a, ddouble *b);
