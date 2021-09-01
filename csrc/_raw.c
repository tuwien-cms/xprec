#include "Python.h"
#include "math.h"
#include "stdbool.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"


/**
 * Type for double-double calculations
 */
typedef struct {
    double hi;
    double lo;
} ddouble;


/**
 * Allows parameter to be marked unused
 */
#define MARK_UNUSED(x)  do { (void)(x); } while(false)

/**
 * Create ufunc loop routine for a unary operation
 */
#define UNARY_FUNCTION(func_name, inner_func, type_out, type_in)        \
    static void func_name(char **args, const npy_intp *dimensions,      \
                          const npy_intp *steps, void *data)            \
    {                                                                   \
        assert (sizeof(ddouble) == 2 * sizeof(double));                 \
        npy_intp i;                                                     \
        npy_intp n = dimensions[0];                                     \
        char *_in1 = args[0], *_out1 = args[1];                         \
        npy_intp is1 = steps[0], os1 = steps[1];                        \
                                                                        \
        for (i = 0; i < n; i++) {                                       \
            const type_in *in = (const type_in *)_in1;                  \
            type_out *out = (type_out *)_out1;                          \
            *out = inner_func(*in);                                     \
                                                                        \
            _in1 += is1;                                                \
            _out1 += os1;                                               \
        }                                                               \
        MARK_UNUSED(data);                                              \
    }

/**
 * Create ufunc loop routine for a binary operation
 */
#define BINARY_FUNCTION(func_name, inner_func, type_r, type_a, type_b)  \
    static void func_name(char **args, const npy_intp *dimensions,      \
                          const npy_intp* steps, void *data)            \
    {                                                                   \
        assert (sizeof(ddouble) == 2 * sizeof(double));                 \
        npy_intp i;                                                     \
        npy_intp n = dimensions[0];                                     \
        char *_in1 = args[0], *_in2 = args[1], *_out1 = args[2];        \
        npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];        \
                                                                        \
        for (i = 0; i < n; i++) {                                       \
            const type_a *lhs = (const type_a *)_in1;                   \
            const type_b *rhs = (const type_b *)_in2;                   \
            type_r *out = (type_r *)_out1;                              \
            *out = inner_func(*lhs, *rhs);                              \
                                                                        \
            _in1 += is1;                                                \
            _in2 += is2;                                                \
            _out1 += os1;                                               \
        }                                                               \
        MARK_UNUSED(data);                                              \
    }


/* ----------------------- Functions ----------------------------- */

inline ddouble two_sum_quick(double a, double b)
{
    double s = a + b;
    double lo = b - (s - a);
    return (ddouble){.hi = s, .lo = lo};
}

inline ddouble two_sum(double a, double b)
{
    double s = a + b;
    double v = s - a;
    double lo = (a - (s - v)) + (b - v);
    return (ddouble){.hi = s, .lo = lo};
}

inline ddouble two_diff(double a, double b)
{
    double s = a - b;
    double v = s - a;
    double lo = (a - (s - v)) - (b + v);
    return (ddouble){.hi = s, .lo = lo};
}

inline ddouble two_prod(double a, double b)
{
    double s = a * b;
    double lo = fma(a, b, -s);
    return (ddouble){.hi = s, .lo = lo};
}

/* -------------------- Combining quad/double ------------------------ */

inline ddouble addqd(ddouble x, double y)
{
    ddouble s = two_sum(x.hi, y);
    double v = x.lo + s.lo;
    return two_sum_quick(s.hi, v);
}
BINARY_FUNCTION(u_addqd, addqd, ddouble, ddouble, double)

inline ddouble subqd(ddouble x, double y)
{
    ddouble s = two_diff(x.hi, y);
    double v = x.lo + s.lo;
    return two_sum_quick(s.hi, v);
}
BINARY_FUNCTION(u_subqd, subqd, ddouble, ddouble, double)

inline ddouble mulqd(ddouble x, double y)
{
    ddouble c = two_prod(x.hi, y);
    double v = fma(x.lo, y, c.lo);
    return two_sum_quick(c.hi, v);
}
BINARY_FUNCTION(u_mulqd, mulqd, ddouble, ddouble, double)

inline ddouble divqd(ddouble x, double y)
{
    /* Alg 14 */
    double t_hi = x.hi / y;
    ddouble pi = two_prod(t_hi, y);
    double d_hi = x.hi - pi.hi;
    double d_lo = x.lo - pi.lo;
    double t_lo = (d_hi + d_lo) / y;
    return two_sum_quick(t_hi, t_lo);
}
BINARY_FUNCTION(u_divqd, divqd, ddouble, ddouble, double)

/* -------------------- Combining double/quad ------------------------- */

ddouble negq(ddouble);
ddouble invq(ddouble);

inline ddouble adddq(double x, ddouble y)
{
    return addqd(y, x);
}
BINARY_FUNCTION(u_adddq, adddq, ddouble, double, ddouble)

inline ddouble subdq(double x, ddouble y)
{
    /* TODO: Probably not ideal */
    return addqd(negq(y), x);
}
BINARY_FUNCTION(u_subdq, subdq, ddouble, double, ddouble)

inline ddouble muldq(double x, ddouble y)
{
    return mulqd(y, x);
}
BINARY_FUNCTION(u_muldq, muldq, ddouble, double, ddouble)

inline ddouble divdq(double x, ddouble y)
{
    /* TODO: Probably not ideal */
    return mulqd(invq(y), x);
}
BINARY_FUNCTION(u_divdq, divdq, ddouble, double, ddouble)

inline ddouble mul_pwr2(ddouble a, double b) {
    return (ddouble){a.hi * b, a.lo * b};
}

/* -------------------- Combining quad/quad ------------------------- */

inline ddouble addqq(ddouble x, ddouble y)
{
    ddouble s = two_sum(x.hi, y.hi);
    ddouble t = two_sum(x.lo, y.lo);
    ddouble v = two_sum_quick(s.hi, s.lo + t.hi);
    ddouble z = two_sum_quick(v.hi, t.lo + v.lo);
    return z;
}
BINARY_FUNCTION(u_addqq, addqq, ddouble, ddouble, ddouble)

inline ddouble subqq(ddouble x, ddouble y)
{
    ddouble s = two_diff(x.hi, y.hi);
    ddouble t = two_diff(x.lo, y.lo);
    ddouble v = two_sum_quick(s.hi, s.lo + t.hi);
    ddouble z = two_sum_quick(v.hi, t.lo + v.lo);
    return z;
}
BINARY_FUNCTION(u_subqq, subqq, ddouble, ddouble, ddouble)

inline ddouble mulqq(ddouble a, ddouble b)
{
    /* Alg 11 */
    ddouble c = two_prod(a.hi, b.hi);
    double t = a.hi * b.lo;
    t = fma(a.lo, b.hi, t);
    return two_sum_quick(c.hi, c.lo + t);
}
BINARY_FUNCTION(u_mulqq, mulqq, ddouble, ddouble, ddouble)

inline ddouble divqq(ddouble x, ddouble y)
{
    /* Alg 17 */
    double t_hi = x.hi / y.hi;
    ddouble r = mulqd(y, t_hi);
    double pi_hi = x.hi - r.hi;
    double d = pi_hi + (x.lo - r.lo);
    double t_lo = d / y.hi;
    return two_sum_quick(t_hi, t_lo);
}
BINARY_FUNCTION(u_divqq, divqq, ddouble, ddouble, ddouble)

/* -------------------- Unary functions ------------------------- */

inline ddouble negq(ddouble a)
{
    return (ddouble){-a.hi, -a.lo};
}
UNARY_FUNCTION(u_negq, negq, ddouble, ddouble)

inline ddouble posq(ddouble a)
{
    return (ddouble){-a.hi, -a.lo};
}
UNARY_FUNCTION(u_posq, posq, ddouble, ddouble)

inline ddouble absq(ddouble a)
{
    return signbit(a.hi) ? negq(a) : a;
}
UNARY_FUNCTION(u_absq, absq, ddouble, ddouble)

inline ddouble invq(ddouble y)
{
    /* Alg 17 with x = 1 */
    double t_hi = 1.0 / y.hi;
    ddouble r = mulqd(y, t_hi);
    double pi_hi = 1.0 - r.hi;
    double d = pi_hi - r.lo;
    double t_lo = d / y.hi;
    return two_sum_quick(t_hi, t_lo);
}
UNARY_FUNCTION(u_invq, invq, ddouble, ddouble)

inline ddouble sqrq(ddouble a)
{
    /* Alg 11 */
    ddouble c = two_prod(a.hi, a.hi);
    double t = 2 * a.hi * a.lo;
    return two_sum_quick(c.hi, c.lo + t);
}
UNARY_FUNCTION(u_sqrq, sqrq, ddouble, ddouble)

inline ddouble roundq(ddouble a)
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
UNARY_FUNCTION(u_roundq, roundq, ddouble, ddouble)

inline ddouble floorq(ddouble a)
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
UNARY_FUNCTION(u_floorq, floorq, ddouble, ddouble)

inline ddouble ceilq(ddouble a)
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
UNARY_FUNCTION(u_ceilq, ceilq, ddouble, ddouble)

inline ddouble dremq(ddouble a, ddouble b)
{
    ddouble n = roundq(divqq(a, b));
    return subqq(a, mulqq(n, b));
}

inline ddouble divremq(ddouble a, ddouble b, ddouble *r)
{
    ddouble n = roundq(divqq(a, b));
    *r = subqq(a, mulqq(n, b));
    return n;
}

/******************************** Constants *********************************/

inline ddouble nanq()
{
    double nan = strtod("NaN", NULL);
    return (ddouble){nan, nan};
}

inline ddouble infq()
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
static const ddouble Q_E = {2.718281828459045091e+00, 1.445646891729250158e-16};
static const ddouble Q_LOG2 = {6.931471805599452862e-01, 2.319046813846299558e-17};
static const ddouble Q_LOG10 = {2.302585092994045901e+00, -2.170756223382249351e-16};

static const ddouble Q_EPS = {4.93038065763132e-32, 0.0};
static const ddouble Q_MIN = {2.0041683600089728e-292, 0.0};
static const ddouble Q_MAX =
    {1.79769313486231570815e+308, 9.97920154767359795037e+291};

/*********************** Comparisons q/q ***************************/

inline bool equalqq(ddouble a, ddouble b)
{
    return a.hi == b.hi && a.lo == b.lo;
}
BINARY_FUNCTION(u_equalqq, equalqq, bool, ddouble, ddouble)

inline bool notequalqq(ddouble a, ddouble b)
{
    return a.hi != b.hi || a.lo != b.lo;
}
BINARY_FUNCTION(u_notequalqq, notequalqq, bool, ddouble, ddouble)

inline bool greaterqq(ddouble a, ddouble b)
{
    return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo);
}
BINARY_FUNCTION(u_greaterqq, greaterqq, bool, ddouble, ddouble)

inline bool lessqq(ddouble a, ddouble b)
{
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
BINARY_FUNCTION(u_lessqq, lessqq, bool, ddouble, ddouble)

inline bool greaterequalqq(ddouble a, ddouble b)
{
    return a.hi > b.hi || (a.hi == b.hi && a.lo >= b.lo);
}
BINARY_FUNCTION(u_greaterequalqq, greaterqq, bool, ddouble, ddouble)

inline bool lessequalqq(ddouble a, ddouble b)
{
    return a.hi < b.hi || (a.hi == b.hi && a.lo <= b.lo);
}
BINARY_FUNCTION(u_lessequalqq, lessqq, bool, ddouble, ddouble)

/*********************** Comparisons q/d ***************************/

inline bool equalqd(ddouble a, double b)
{
    return equalqq(a, (ddouble){b, 0});
}
BINARY_FUNCTION(u_equalqd, equalqd, bool, ddouble, double)

inline bool notequalqd(ddouble a, double b)
{
    return notequalqq(a, (ddouble){b, 0});
}
BINARY_FUNCTION(u_notequalqd, notequalqd, bool, ddouble, double)

inline bool greaterqd(ddouble a, double b)
{
    return greaterqq(a, (ddouble){b, 0});
}
BINARY_FUNCTION(u_greaterqd, greaterqd, bool, ddouble, double)

inline bool lessqd(ddouble a, double b)
{
    return lessqq(a, (ddouble){b, 0});
}
BINARY_FUNCTION(u_lessqd, lessqd, bool, ddouble, double)

inline bool greaterequalqd(ddouble a, double b)
{
    return greaterequalqq(a, (ddouble){b, 0});
}
BINARY_FUNCTION(u_greaterequalqd, greaterqd, bool, ddouble, double)

inline bool lessequalqd(ddouble a, double b)
{
    return lessequalqq(a, (ddouble){b, 0});
}
BINARY_FUNCTION(u_lessequalqd, lessqd, bool, ddouble, double)

/*********************** Comparisons d/q ***************************/

inline bool equaldq(double a, ddouble b)
{
    return equalqq((ddouble){a, 0}, b);
}
BINARY_FUNCTION(u_equaldq, equaldq, bool, double, ddouble)

inline bool notequaldq(double a, ddouble b)
{
    return notequalqq((ddouble){a, 0}, b);
}
BINARY_FUNCTION(u_notequaldq, notequaldq, bool, double, ddouble)

inline bool greaterdq(double a, ddouble b)
{
    return greaterqq((ddouble){a, 0}, b);
}
BINARY_FUNCTION(u_greaterdq, greaterdq, bool, double, ddouble)

inline bool lessdq(double a, ddouble b)
{
    return lessqq((ddouble){a, 0}, b);
}
BINARY_FUNCTION(u_lessdq, lessdq, bool, double, ddouble)

inline bool greaterequaldq(double a, ddouble b)
{
    return greaterequalqq((ddouble){a, 0}, b);
}
BINARY_FUNCTION(u_greaterequaldq, greaterdq, bool, double, ddouble)

inline bool lessequaldq(double a, ddouble b)
{
    return lessequalqq((ddouble){a, 0}, b);
}
BINARY_FUNCTION(u_lessequaldq, lessdq, bool, double, ddouble)

/************************** Unary tests **************************/

inline bool iszeroq(ddouble x)
{
    return x.hi == 0.0;
}
UNARY_FUNCTION(u_iszeroq, iszeroq, bool, ddouble)

inline bool isoneq(ddouble x)
{
    return x.hi == 1.0 && x.lo == 0.0;
}
UNARY_FUNCTION(u_isoneq, isoneq, bool, ddouble)

inline bool ispositiveq(ddouble x)
{
    return x.hi > 0.0;
}
UNARY_FUNCTION(u_ispositiveq, ispositiveq, bool, ddouble)

inline bool isnegativeq(ddouble x)
{
    return x.hi < 0.0;
}
UNARY_FUNCTION(u_isnegativeq, isnegativeq, bool, ddouble)

/************************** Advanced math functions ********************/

inline ddouble sqrtq(ddouble a)
{
    /* Algorithm from Karp (QD library) */
    if (a.hi <= 0)
        return (ddouble){.hi = sqrt(a.hi), .lo = 0};

    double x = 1.0 / sqrt(a.hi);
    double ax = a.hi * x;

    ddouble ax_sqr = sqrq((ddouble){ax, 0});
    double diff = subqq(a, ax_sqr).hi * x * 0.5;
    return two_sum(ax, diff);
}
UNARY_FUNCTION(u_sqrtq, sqrtq, ddouble, ddouble)

inline ddouble ldexpq(ddouble a, int exp)
{
    return (ddouble) {ldexp(a.hi, exp), ldexp(a.lo, exp)};
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
UNARY_FUNCTION(u_expq, expq, ddouble, ddouble)

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
UNARY_FUNCTION(u_expm1q, expm1q, ddouble, ddouble)

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
UNARY_FUNCTION(u_logq, logq, ddouble, ddouble)

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
    if (iszeroq(a))
    {
        *sin_a = Q_ZERO;
        *cos_a = Q_ONE;
        return;
    }

    *sin_a = sin_taylor(a);
    *cos_a = sqrtq(subdq(1.0, sqrq(*sin_a)));
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
UNARY_FUNCTION(u_sinq, sinq, ddouble, ddouble)

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
UNARY_FUNCTION(u_cosq, cosq, ddouble, ddouble)

ddouble sinhq(ddouble a)
{
    if (iszeroq(a))
        return Q_ZERO;

    if (absq(a).hi > 0.05) {
        ddouble ea = expq(a);
        return mul_pwr2(subqq(ea, invq(ea)), 0.5);
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
UNARY_FUNCTION(u_sinhq, sinhq, ddouble, ddouble)

ddouble coshq(ddouble a)
{
    if (iszeroq(a))
        return Q_ONE;

    ddouble ea = expq(a);
    return mul_pwr2(addqq(ea, invq(ea)), 0.5);
}
UNARY_FUNCTION(u_coshq, coshq, ddouble, ddouble)

ddouble tanhq(ddouble a)
{
    if (iszeroq(a))
        return Q_ZERO;

    if (fabs(a.hi) > 0.05) {
        ddouble ea = expq(a);
        ddouble inv_ea = invq(ea);
        return divqq(subqq(ea, inv_ea), addqq(ea, inv_ea));
    }

    ddouble s, c;
    s = sinhq(a);
    c = sqrtq(adddq(1.0, sqrq(s)));
    return divqq(s, c);
}
UNARY_FUNCTION(u_tanhq, tanhq, ddouble, ddouble)

/* ----------------------- Python stuff -------------------------- */

const char DDOUBLE_WRAP = NPY_CDOUBLE;

static void binary_ufunc(PyObject *module_dict, PyUFuncGenericFunction dq_func,
        PyUFuncGenericFunction qd_func, PyUFuncGenericFunction qq_func,
        char ret_dtype, const char *name, const char *docstring)
{

    PyObject *ufunc;
    PyUFuncGenericFunction* loops = PyMem_New(PyUFuncGenericFunction, 3);
    char *dtypes = PyMem_New(char, 3 * 3);
    void **data = PyMem_New(void *, 3);

    loops[0] = dq_func;
    data[0] = NULL;
    dtypes[0] = NPY_DOUBLE;
    dtypes[1] = DDOUBLE_WRAP;
    dtypes[2] = ret_dtype;

    loops[1] = qd_func;
    data[1] = NULL;
    dtypes[3] = DDOUBLE_WRAP;
    dtypes[4] = NPY_DOUBLE;
    dtypes[5] = ret_dtype;

    loops[2] = qq_func;
    data[2] = NULL;
    dtypes[6] = DDOUBLE_WRAP;
    dtypes[7] = DDOUBLE_WRAP;
    dtypes[8] = ret_dtype;

    ufunc = PyUFunc_FromFuncAndData(
                loops, data, dtypes, 3, 2, 1, PyUFunc_None, name, docstring, 0);
    PyDict_SetItemString(module_dict, name, ufunc);
    Py_DECREF(ufunc);
}

static void unary_ufunc(PyObject *module_dict,
                        PyUFuncGenericFunction func, char ret_dtype,
                        const char *name, const char *docstring)
{
    PyObject *ufunc;
    PyUFuncGenericFunction* loops = PyMem_New(PyUFuncGenericFunction, 1);
    char *dtypes = PyMem_New(char, 1 * 2);
    void **data = PyMem_New(void *, 1);

    loops[0] = func;
    data[0] = NULL;
    dtypes[0] = DDOUBLE_WRAP;
    dtypes[1] = ret_dtype;

    ufunc = PyUFunc_FromFuncAndData(
                loops, data, dtypes, 1, 1, 1, PyUFunc_None, name, docstring, 0);
    PyDict_SetItemString(module_dict, name, ufunc);
    Py_DECREF(ufunc);
}

static void constant(PyObject *module_dict, ddouble value, const char *name)
{
    // Note that data must be allocated using malloc, not python allocators!
    ddouble *data = malloc(sizeof value);
    *data = value;

    PyArrayObject *array = (PyArrayObject *)
        PyArray_SimpleNewFromData(0, NULL, DDOUBLE_WRAP, data);
    PyArray_ENABLEFLAGS(array, NPY_ARRAY_OWNDATA);
    PyArray_CLEARFLAGS(array, NPY_ARRAY_WRITEABLE);

    PyDict_SetItemString(module_dict, name, (PyObject *)array);
    Py_DECREF(array);
}

PyMODINIT_FUNC PyInit__raw(void)
{
    // Defitions
    static PyMethodDef no_methods[] = {
        {NULL, NULL, 0, NULL}    // No methods defined
    };
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_raw",
        NULL,
        -1,
        no_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };

    /* Module definition */
    PyObject *module, *module_dict;
    PyArray_Descr *dtype;

    /* Create module */
    module = PyModule_Create(&module_def);
    if (!module)
        return NULL;
    module_dict = PyModule_GetDict(module);

    /* Initialize numpy things */
    import_array();
    import_umath();

    /* Create ufuncs */
    binary_ufunc(module_dict, u_adddq, u_addqd, u_addqq,
                 DDOUBLE_WRAP, "add", "addition");
    binary_ufunc(module_dict, u_subdq, u_subqd, u_subqq,
                 DDOUBLE_WRAP, "subtract", "subtraction");
    binary_ufunc(module_dict, u_muldq, u_mulqd, u_mulqq,
                 DDOUBLE_WRAP, "multiply", "element-wise multiplication");
    binary_ufunc(module_dict, u_divdq, u_divqd, u_divqq,
                 DDOUBLE_WRAP, "true_divide", "element-wise division");

    binary_ufunc(module_dict, u_equaldq, u_equalqd, u_equalqq,
                 NPY_BOOL, "equal", "equality comparison");
    binary_ufunc(module_dict, u_notequaldq, u_notequalqd, u_notequalqq,
                 NPY_BOOL, "not_equal", "inequality comparison");
    binary_ufunc(module_dict, u_greaterdq, u_greaterqd, u_greaterqq,
                 NPY_BOOL, "greater", "element-wise greater");
    binary_ufunc(module_dict, u_lessdq, u_lessqd, u_lessqq,
                 NPY_BOOL, "less", "element-wise less");
    binary_ufunc(module_dict, u_greaterequaldq, u_greaterequalqd, u_greaterequalqq,
                 NPY_BOOL, "greater_equal", "element-wise greater or equal");
    binary_ufunc(module_dict, u_lessequaldq, u_lessequalqd, u_lessequalqq,
                 NPY_BOOL, "less_equal", "element-wise less or equal");

    unary_ufunc(module_dict, u_negq, DDOUBLE_WRAP,
                "negative", "negation (+ to -)");
    unary_ufunc(module_dict, u_posq, DDOUBLE_WRAP,
                "positive", "explicit + sign");
    unary_ufunc(module_dict, u_absq, DDOUBLE_WRAP,
                "absolute", "absolute value");
    unary_ufunc(module_dict, u_invq, DDOUBLE_WRAP,
                "invert", "reciprocal value");
    unary_ufunc(module_dict, u_sqrq, DDOUBLE_WRAP,
                "square", "element-wise square");
    unary_ufunc(module_dict, u_sqrtq, DDOUBLE_WRAP,
                "sqrt", "element-wise square root");

    unary_ufunc(module_dict, u_roundq, DDOUBLE_WRAP,
                "round", "round to nearest integer");
    unary_ufunc(module_dict, u_floorq, DDOUBLE_WRAP,
                "floor", "round down to next integer");
    unary_ufunc(module_dict, u_ceilq, DDOUBLE_WRAP,
                "ceil", "round up to next integer");
    unary_ufunc(module_dict, u_expq, DDOUBLE_WRAP,
                "exp", "exponential function");
    unary_ufunc(module_dict, u_expm1q, DDOUBLE_WRAP,
                "expm1", "exponential function minus one");
    unary_ufunc(module_dict, u_logq, DDOUBLE_WRAP,
                "log", "natural logarithm");
    unary_ufunc(module_dict, u_sinq, DDOUBLE_WRAP,
                "sin", "sine");
    unary_ufunc(module_dict, u_cosq, DDOUBLE_WRAP,
                "cos", "cosine");
    unary_ufunc(module_dict, u_sinhq, DDOUBLE_WRAP,
                "sinh", "hyperbolic sine");
    unary_ufunc(module_dict, u_coshq, DDOUBLE_WRAP,
                "cosh", "hyperbolic cosine");
    unary_ufunc(module_dict, u_tanhq, DDOUBLE_WRAP,
                "tanh", "hyperbolic tangent");

    unary_ufunc(module_dict, u_iszeroq, NPY_BOOL,
                "iszero", "element-wise test for zero");
    unary_ufunc(module_dict, u_isoneq, NPY_BOOL,
                "isone", "element-wise test for one");
    unary_ufunc(module_dict, u_ispositiveq, NPY_BOOL,
                "ispositive", "element-wise test for positive values");
    unary_ufunc(module_dict, u_isnegativeq, NPY_BOOL,
                "isnegative", "element-wise test for negative values");

    constant(module_dict, Q_MAX, "MAX");
    constant(module_dict, Q_MIN, "MIN");
    constant(module_dict, Q_EPS, "EPS");
    constant(module_dict, Q_2PI, "TWOPI");
    constant(module_dict, Q_PI, "PI");
    constant(module_dict, Q_PI_2, "PI_2");
    constant(module_dict, Q_PI_4, "PI_4");
    constant(module_dict, Q_E, "E");
    constant(module_dict, Q_LOG2, "LOG2");
    constant(module_dict, Q_LOG10, "LOG10");

    /* Make dtype */
    dtype = PyArray_DescrFromType(DDOUBLE_WRAP);
    PyDict_SetItemString(module_dict, "dtype", (PyObject *)dtype);

    /* Module is ready */
    return module;
}
