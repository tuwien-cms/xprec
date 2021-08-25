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
 * Create ufunc loop routine for a unary operation
 */
#define DDOUBLE_UNARY_FUNCTION(func_name, inner_func)                   \
    static void func_name(char **args, const npy_intp *dimensions,      \
                          const npy_intp* steps, void* data)            \
    {                                                                   \
        assert (sizeof(ddouble) == 2 * sizeof(double));                 \
        npy_intp i;                                                     \
        npy_intp n = dimensions[0];                                     \
        char *_in1 = args[0], *_out1 = args[1];                         \
        npy_intp is1 = steps[0], os1 = steps[1];                        \
                                                                        \
        for (i = 0; i < n; i++) {                                       \
            const ddouble *in = (const ddouble *)_in1;                  \
            ddouble *out = (ddouble *)_out1;                            \
            *out = inner_func(*in);                                     \
                                                                        \
            _in1 += is1;                                                \
            _out1 += os1;                                               \
        }                                                               \
    }

/**
 * Create ufunc loop routine for a binary operation
 */
#define BINARY_FUNCTION(func_name, inner_func, type_r, type_a, type_b)  \
    static void func_name(char **args, const npy_intp *dimensions,      \
                          const npy_intp* steps, void* data)            \
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
DDOUBLE_UNARY_FUNCTION(u_negq, negq)

inline ddouble posq(ddouble a)
{
    return (ddouble){-a.hi, -a.lo};
}
DDOUBLE_UNARY_FUNCTION(u_posq, posq)

inline ddouble absq(ddouble a)
{
    return signbit(a.hi) ? negq(a) : a;
}
DDOUBLE_UNARY_FUNCTION(u_absq, absq)

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
DDOUBLE_UNARY_FUNCTION(u_invq, invq)

inline ddouble sqrq(ddouble a)
{
    /* Alg 11 */
    ddouble c = two_prod(a.hi, a.hi);
    double t = 2 * a.hi * a.lo;
    return two_sum_quick(c.hi, c.lo + t);
}
DDOUBLE_UNARY_FUNCTION(u_sqrq, sqrq)

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
DDOUBLE_UNARY_FUNCTION(u_sqrtq, sqrtq)





/* ----------------------- Python stuff -------------------------- */

static PyArray_Descr *make_ddouble_dtype()
{
    PyObject *dtype_tuple;
    PyArray_Descr *dtype;

    dtype_tuple = Py_BuildValue("[(s, s), (s, s)]", "hi", "d", "lo", "d");
    PyArray_DescrConverter(dtype_tuple, &dtype);
    Py_DECREF(dtype_tuple);
    return dtype;
}

static void binary_ufunc(PyArray_Descr *q_dtype, PyObject *module_dict,
        PyUFuncGenericFunction dq_func,
        PyUFuncGenericFunction qd_func, PyUFuncGenericFunction qq_func,
        const char *name, const char *docstring)
{
    PyObject *ufunc;
    PyArray_Descr *d_dtype = PyArray_DescrFromType(NPY_DOUBLE);
    PyArray_Descr *dq_dtypes[] = {d_dtype, q_dtype, q_dtype},
                  *qd_dtypes[] = {q_dtype, d_dtype, q_dtype},
                  *qq_dtypes[] = {q_dtype, q_dtype, q_dtype};

    ufunc = PyUFunc_FromFuncAndData(
                NULL, NULL, NULL, 0, 2, 1, PyUFunc_None, name, docstring, 0);
    PyUFunc_RegisterLoopForDescr(
                (PyUFuncObject *)ufunc, q_dtype, dq_func, dq_dtypes, NULL);
    PyUFunc_RegisterLoopForDescr(
                (PyUFuncObject *)ufunc, q_dtype, qd_func, qd_dtypes, NULL);
    PyUFunc_RegisterLoopForDescr(
                (PyUFuncObject *)ufunc, q_dtype, qq_func, qq_dtypes, NULL);
    PyDict_SetItemString(module_dict, name, ufunc);

    Py_DECREF(ufunc);
    Py_DECREF(d_dtype);
}

static void ddouble_ufunc(PyArray_Descr *dtype, PyObject *module_dict,
                          PyUFuncGenericFunction func, int nargs,
                          const char *name, const char *docstring)
{
    PyObject *ufunc;
    PyArray_Descr *dtypes[] = {dtype, dtype, dtype};

    ufunc = PyUFunc_FromFuncAndData(
                NULL, NULL, NULL, 0, nargs, 1, PyUFunc_None, name, docstring, 0);
    PyUFunc_RegisterLoopForDescr(
                (PyUFuncObject *)ufunc, dtype, func, dtypes, NULL);
    PyDict_SetItemString(module_dict, name, ufunc);
    Py_DECREF(ufunc);
}

// Init routine
PyMODINIT_FUNC PyInit__ddouble(void)
{
    static PyMethodDef no_methods[] = {
        {NULL, NULL, 0, NULL}    // No methods defined
    };
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_ddouble",
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

    /* Build ufunc dtype */
    dtype = make_ddouble_dtype(module_dict);

    /* Create ufuncs */
    //ddouble_ufunc(dtype, module_dict, u_addqq, 2, "add", "");
    binary_ufunc(dtype, module_dict, u_adddq, u_addqd, u_addqq, "add", "");
    binary_ufunc(dtype, module_dict, u_subdq, u_subqd, u_subqq, "sub", "");
    binary_ufunc(dtype, module_dict, u_muldq, u_mulqd, u_mulqq, "mul", "");
    binary_ufunc(dtype, module_dict, u_divdq, u_divqd, u_divqq, "div", "");

    ddouble_ufunc(dtype, module_dict, u_negq, 1, "neg", "");
    ddouble_ufunc(dtype, module_dict, u_posq, 1, "pos", "");
    ddouble_ufunc(dtype, module_dict, u_absq, 1, "abs", "");
    ddouble_ufunc(dtype, module_dict, u_invq, 1, "inv", "");
    ddouble_ufunc(dtype, module_dict, u_sqrq, 1, "sqr", "");
    ddouble_ufunc(dtype, module_dict, u_sqrtq, 1, "sqrt", "");

    /* Store dtype in module and return */
    PyDict_SetItemString(module_dict, "dtype", (PyObject *)dtype);
    return module;
}
