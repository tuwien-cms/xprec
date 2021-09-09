#include "Python.h"
#include "math.h"
#include "stdio.h"

#include "dd_arith.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

/**
 * Allows parameter to be marked unused
 */
#define MARK_UNUSED(x)  do { (void)(x); } while(false)

/**
 * Create ufunc loop routine for a unary operation
 */
#define ULOOP_UNARY(func_name, inner_func, type_out, type_in)           \
    static void func_name(char **args, const npy_intp *dimensions,      \
                          const npy_intp *steps, void *data)            \
    {                                                                   \
        const npy_intp n = dimensions[0];                               \
        const npy_intp is = steps[0] / sizeof(type_in),                 \
                       os = steps[1] / sizeof(type_out);                \
        const type_in *in = (const type_in *)args[0];                   \
        type_out *out = (type_out *)args[1];                            \
                                                                        \
        _Pragma("omp parallel for")                                     \
        for (npy_intp i = 0; i < n; ++i)                                \
            out[i * os] = inner_func(in[i * is]);                       \
        MARK_UNUSED(data);                                              \
    }

/**
 * Create ufunc loop routine for a binary operation
 */
#define ULOOP_BINARY(func_name, inner_func, type_out, type_a, type_b)   \
    static void func_name(char **args, const npy_intp *dimensions,      \
                          const npy_intp* steps, void *data)            \
    {                                                                   \
        const npy_intp n = dimensions[0];                               \
        const npy_intp as = steps[0] / sizeof(type_a),                  \
                       bs = steps[1] / sizeof(type_b),                  \
                       os = steps[2] / sizeof(type_out);                \
        const type_a *a = (const type_a *)args[0];                      \
        const type_b *b = (const type_b *)args[1];                      \
        type_out *out = (type_out *)args[2];                            \
                                                                        \
        _Pragma("omp parallel for")                                     \
        for (npy_intp i = 0; i < n; ++i)                                \
            out[i * os] = inner_func(a[i * as], b[i * bs]);             \
        MARK_UNUSED(data);                                              \
    }

ULOOP_BINARY(u_addqd, addqd, ddouble, ddouble, double)
ULOOP_BINARY(u_subqd, subqd, ddouble, ddouble, double)
ULOOP_BINARY(u_mulqd, mulqd, ddouble, ddouble, double)
ULOOP_BINARY(u_divqd, divqd, ddouble, ddouble, double)
ULOOP_BINARY(u_adddq, adddq, ddouble, double, ddouble)
ULOOP_BINARY(u_subdq, subdq, ddouble, double, ddouble)
ULOOP_BINARY(u_muldq, muldq, ddouble, double, ddouble)
ULOOP_BINARY(u_divdq, divdq, ddouble, double, ddouble)

ULOOP_BINARY(u_addqq, addqq, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_subqq, subqq, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_mulqq, mulqq, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_divqq, divqq, ddouble, ddouble, ddouble)

ULOOP_UNARY(u_negq, negq, ddouble, ddouble)
ULOOP_UNARY(u_posq, posq, ddouble, ddouble)
ULOOP_UNARY(u_absq, absq, ddouble, ddouble)
ULOOP_UNARY(u_reciprocalq, reciprocalq, ddouble, ddouble)
ULOOP_UNARY(u_sqrq, sqrq, ddouble, ddouble)
ULOOP_UNARY(u_roundq, roundq, ddouble, ddouble)
ULOOP_UNARY(u_floorq, floorq, ddouble, ddouble)
ULOOP_UNARY(u_ceilq, ceilq, ddouble, ddouble)

ULOOP_UNARY(u_signbitq, signbitq, bool, ddouble)
ULOOP_BINARY(u_copysignqq, copysignqq, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_copysignqd, copysignqd, ddouble, ddouble, double)
ULOOP_BINARY(u_copysigndq, copysigndq, ddouble, double, ddouble)
ULOOP_UNARY(u_signq, signq, ddouble, ddouble)

ULOOP_UNARY(u_isfiniteq, isfiniteq, bool, ddouble)
ULOOP_UNARY(u_isinfq, isinfq, bool, ddouble)
ULOOP_UNARY(u_isnanq, isnanq, bool, ddouble)

ULOOP_BINARY(u_equalqq, equalqq, bool, ddouble, ddouble)
ULOOP_BINARY(u_notequalqq, notequalqq, bool, ddouble, ddouble)
ULOOP_BINARY(u_greaterqq, greaterqq, bool, ddouble, ddouble)
ULOOP_BINARY(u_lessqq, lessqq, bool, ddouble, ddouble)
ULOOP_BINARY(u_greaterequalqq, greaterqq, bool, ddouble, ddouble)
ULOOP_BINARY(u_lessequalqq, lessqq, bool, ddouble, ddouble)

ULOOP_BINARY(u_equalqd, equalqd, bool, ddouble, double)
ULOOP_BINARY(u_notequalqd, notequalqd, bool, ddouble, double)
ULOOP_BINARY(u_greaterqd, greaterqd, bool, ddouble, double)
ULOOP_BINARY(u_lessqd, lessqd, bool, ddouble, double)
ULOOP_BINARY(u_greaterequalqd, greaterequalqd, bool, ddouble, double)
ULOOP_BINARY(u_lessequalqd, lessequalqd, bool, ddouble, double)

ULOOP_BINARY(u_equaldq, equaldq, bool, double, ddouble)
ULOOP_BINARY(u_notequaldq, notequaldq, bool, double, ddouble)
ULOOP_BINARY(u_greaterdq, greaterdq, bool, double, ddouble)
ULOOP_BINARY(u_lessdq, lessdq, bool, double, ddouble)
ULOOP_BINARY(u_greaterequaldq, greaterequaldq, bool, double, ddouble)
ULOOP_BINARY(u_lessequaldq, lessequaldq, bool, double, ddouble)

ULOOP_BINARY(u_fminqq, fminqq, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_fmaxqq, fmaxqq, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_fminqd, fminqd, ddouble, ddouble, double)
ULOOP_BINARY(u_fmaxqd, fmaxqd, ddouble, ddouble, double)
ULOOP_BINARY(u_fmindq, fmindq, ddouble, double, ddouble)
ULOOP_BINARY(u_fmaxdq, fmaxdq, ddouble, double, ddouble)

ULOOP_UNARY(u_iszeroq, iszeroq, bool, ddouble)
ULOOP_UNARY(u_isoneq, isoneq, bool, ddouble)
ULOOP_UNARY(u_ispositiveq, ispositiveq, bool, ddouble)
ULOOP_UNARY(u_isnegativeq, isnegativeq, bool, ddouble)

ULOOP_UNARY(u_sqrtq, sqrtq, ddouble, ddouble)
ULOOP_BINARY(u_hypotqq, hypotqq, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_hypotdq, hypotdq, ddouble, double, ddouble)
ULOOP_BINARY(u_hypotqd, hypotqd, ddouble, ddouble, double)

ULOOP_UNARY(u_expq, expq, ddouble, ddouble)
ULOOP_UNARY(u_expm1q, expm1q, ddouble, ddouble)

ULOOP_UNARY(u_logq, logq, ddouble, ddouble)
ULOOP_UNARY(u_sinq, sinq, ddouble, ddouble)
ULOOP_UNARY(u_cosq, cosq, ddouble, ddouble)
ULOOP_UNARY(u_sinhq, sinhq, ddouble, ddouble)
ULOOP_UNARY(u_coshq, coshq, ddouble, ddouble)
ULOOP_UNARY(u_tanhq, tanhq, ddouble, ddouble)

/* ----------------------- Python stuff -------------------------- */

static const char DDOUBLE_WRAP = NPY_CDOUBLE;

static PyObject *module = NULL;

PyObject *make_module()
{
    // Defitions
    static PyMethodDef no_methods[] = {
        {NULL, NULL, 0, NULL}    // No methods defined
    };
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_dd_ufunc",
        NULL,
        -1,
        no_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };

    /* Module definition */
    module = PyModule_Create(&module_def);
    return module;
}

static void binary_ufunc(PyUFuncGenericFunction dq_func,
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
    PyModule_AddObject(module, name, ufunc);
}

static void unary_ufunc(PyUFuncGenericFunction func, char ret_dtype,
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
    PyModule_AddObject(module, name, ufunc);
}

static void constant(ddouble value, const char *name)
{
    // Note that data must be allocated using malloc, not python allocators!
    ddouble *data = malloc(sizeof value);
    *data = value;

    PyArrayObject *array = (PyArrayObject *)
        PyArray_SimpleNewFromData(0, NULL, DDOUBLE_WRAP, data);
    PyArray_ENABLEFLAGS(array, NPY_ARRAY_OWNDATA);
    PyArray_CLEARFLAGS(array, NPY_ARRAY_WRITEABLE);

    PyModule_AddObject(module, name, (PyObject *)array);
}

PyMODINIT_FUNC PyInit__dd_ufunc(void)
{
    /* Initialize module */
    if (!make_module())
        return NULL;

    /* Initialize numpy things */
    import_array();
    import_umath();

    /* Create ufuncs */
    binary_ufunc(u_adddq, u_addqd, u_addqq,
                 DDOUBLE_WRAP, "add", "addition");
    binary_ufunc(u_subdq, u_subqd, u_subqq,
                 DDOUBLE_WRAP, "subtract", "subtraction");
    binary_ufunc(u_muldq, u_mulqd, u_mulqq,
                 DDOUBLE_WRAP, "multiply", "element-wise multiplication");
    binary_ufunc(u_divdq, u_divqd, u_divqq,
                 DDOUBLE_WRAP, "true_divide", "element-wise division");

    binary_ufunc(u_equaldq, u_equalqd, u_equalqq,
                 NPY_BOOL, "equal", "equality comparison");
    binary_ufunc(u_notequaldq, u_notequalqd, u_notequalqq,
                 NPY_BOOL, "not_equal", "inequality comparison");
    binary_ufunc(u_greaterdq, u_greaterqd, u_greaterqq,
                 NPY_BOOL, "greater", "element-wise greater");
    binary_ufunc(u_lessdq, u_lessqd, u_lessqq,
                 NPY_BOOL, "less", "element-wise less");
    binary_ufunc(u_greaterequaldq, u_greaterequalqd, u_greaterequalqq,
                 NPY_BOOL, "greater_equal", "element-wise greater or equal");
    binary_ufunc(u_lessequaldq, u_lessequalqd, u_lessequalqq,
                 NPY_BOOL, "less_equal", "element-wise less or equal");
    binary_ufunc(u_fmindq, u_fminqd, u_fminqq,
                 DDOUBLE_WRAP, "fmin", "element-wise minimum");
    binary_ufunc(u_fmaxdq, u_fmaxqd, u_fmaxqq,
                 DDOUBLE_WRAP, "fmax", "element-wise minimum");

    unary_ufunc(u_negq, DDOUBLE_WRAP,
                "negative", "negation (+ to -)");
    unary_ufunc(u_posq, DDOUBLE_WRAP,
                "positive", "explicit + sign");
    unary_ufunc(u_absq, DDOUBLE_WRAP,
                "absolute", "absolute value");
    unary_ufunc(u_reciprocalq, DDOUBLE_WRAP,
                "reciprocal", "element-wise reciprocal value");
    unary_ufunc(u_sqrq, DDOUBLE_WRAP,
                "square", "element-wise square");
    unary_ufunc(u_sqrtq, DDOUBLE_WRAP,
                "sqrt", "element-wise square root");
    unary_ufunc(u_signbitq, NPY_BOOL,
                "signbit", "sign bit of number");
    unary_ufunc(u_isfiniteq, NPY_BOOL,
                "isfinite", "whether number is finite");
    unary_ufunc(u_isinfq, NPY_BOOL,
                "isinf", "whether number is infinity");
    unary_ufunc(u_isnanq, NPY_BOOL,
                "isnan", "test for not-a-number");

    unary_ufunc(u_roundq, DDOUBLE_WRAP,
                "rint", "round to nearest integer");
    unary_ufunc(u_floorq, DDOUBLE_WRAP,
                "floor", "round down to next integer");
    unary_ufunc(u_ceilq, DDOUBLE_WRAP,
                "ceil", "round up to next integer");
    unary_ufunc(u_expq, DDOUBLE_WRAP,
                "exp", "exponential function");
    unary_ufunc(u_expm1q, DDOUBLE_WRAP,
                "expm1", "exponential function minus one");
    unary_ufunc(u_logq, DDOUBLE_WRAP,
                "log", "natural logarithm");
    unary_ufunc(u_sinq, DDOUBLE_WRAP,
                "sin", "sine");
    unary_ufunc(u_cosq, DDOUBLE_WRAP,
                "cos", "cosine");
    unary_ufunc(u_sinhq, DDOUBLE_WRAP,
                "sinh", "hyperbolic sine");
    unary_ufunc(u_coshq, DDOUBLE_WRAP,
                "cosh", "hyperbolic cosine");
    unary_ufunc(u_tanhq, DDOUBLE_WRAP,
                "tanh", "hyperbolic tangent");

    unary_ufunc(u_iszeroq, NPY_BOOL,
                "iszero", "element-wise test for zero");
    unary_ufunc(u_isoneq, NPY_BOOL,
                "isone", "element-wise test for one");
    unary_ufunc(u_ispositiveq, NPY_BOOL,
                "ispositive", "element-wise test for positive values");
    unary_ufunc(u_isnegativeq, NPY_BOOL,
                "isnegative", "element-wise test for negative values");
    unary_ufunc(u_signq, DDOUBLE_WRAP,
                "sign", "element-wise sign computation");

    binary_ufunc(u_copysigndq, u_copysignqd, u_copysignqq,
                 DDOUBLE_WRAP, "copysign", "overrides sign of x with that of y");
    binary_ufunc(u_hypotdq, u_hypotqd, u_hypotqq,
                 DDOUBLE_WRAP, "hypot", "hypothenuse calculation");


    constant(Q_MAX, "MAX");
    constant(Q_MIN, "MIN");
    constant(Q_EPS, "EPS");
    constant(Q_2PI, "TWOPI");
    constant(Q_PI, "PI");
    constant(Q_PI_2, "PI_2");
    constant(Q_PI_4, "PI_4");
    constant(Q_E, "E");
    constant(Q_LOG2, "LOG2");
    constant(Q_LOG10, "LOG10");
    constant(nanq(), "NAN");
    constant(infq(), "INF");

    /* Make dtype */
    PyObject *dtype = (PyObject *) PyArray_DescrFromType(DDOUBLE_WRAP);
    PyModule_AddObject(module, "dtype", dtype);

    /* Module is ready */
    return module;
}
