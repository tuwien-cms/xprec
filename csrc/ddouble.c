#include "Python.h"
#include "math.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"


/**
 * Type for double-double calculations
 */
typedef struct {
    double x;
    double e;
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
#define DDOUBLE_BINARY_FUNCTION(func_name, inner_func)                  \
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
            const ddouble *lhs = (const ddouble *)_in1;                 \
            const ddouble *rhs = (const ddouble *)_in2;                 \
            ddouble *out = (ddouble *)_out1;                            \
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
    double e = b - (s - a);
    return (ddouble){.x = s, .e = e};
}

inline ddouble two_sum(double a, double b)
{
    double s = a + b;
    double v = s - a;
    double e = (a - (s - v)) + (b - v);
    return (ddouble){.x = s, .e = e};
}

inline ddouble two_diff(double a, double b)
{
    double s = a - b;
    double v = s - a;
    double e = (a - (s - v)) - (b + v);
    return (ddouble){.x = s, .e = e};
}

inline ddouble two_prod(double a, double b)
{
    double s = a * b;
    double e = fma(a, b, -s);
    return (ddouble){.x = s, .e = e};
}

inline ddouble addq(ddouble a, ddouble b)
{
    ddouble res = two_sum(a.x, b.x);
    res.e += a.e + b.e;
    return two_sum_quick(res.x, res.e);
}
DDOUBLE_BINARY_FUNCTION(u_addq, addq)

inline ddouble subq(ddouble a, ddouble b)
{
    ddouble res = two_diff(a.x, b.x);
    res.e += a.e - b.e;
    return two_sum_quick(res.x, res.e);
}
DDOUBLE_BINARY_FUNCTION(u_subq, subq)

inline ddouble mulq(ddouble a, ddouble b)
{
    ddouble res = two_prod(a.x, b.x);
    res.e += a.x * b.e + a.e * b.x;
    return two_sum_quick(res.x, res.e);
}
DDOUBLE_BINARY_FUNCTION(u_mulq, mulq)

inline ddouble divq(ddouble a, ddouble b)
{
    double r = a.x / b.x;
    ddouble s = two_prod(r, b.x);
    double e = (a.x - s.x - s.e + a.e - r * b.e) / b.x;
    return two_sum_quick(r, e);
}
DDOUBLE_BINARY_FUNCTION(u_divq, divq)

inline ddouble negq(ddouble a)
{
    return (ddouble){-a.x, -a.e};
}
DDOUBLE_UNARY_FUNCTION(u_negq, negq)


/* ----------------------- Python stuff -------------------------- */

static PyArray_Descr *make_ddouble_dtype()
{
    PyObject *dtype_tuple;
    PyArray_Descr *dtype;

    dtype_tuple = Py_BuildValue("[(s, s), (s, s)]", "x", "d", "e", "d");
    PyArray_DescrConverter(dtype_tuple, &dtype);
    Py_DECREF(dtype_tuple);
    return dtype;
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
        "_fma",
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
    ddouble_ufunc(dtype, module_dict, u_addq, 2, "add", "");
    ddouble_ufunc(dtype, module_dict, u_subq, 2, "sub", "");
    ddouble_ufunc(dtype, module_dict, u_mulq, 2, "mul", "");
    ddouble_ufunc(dtype, module_dict, u_divq, 2, "div", "");
    ddouble_ufunc(dtype, module_dict, u_negq, 1, "neg", "");

    /* Store dtype in module and return */
    PyDict_SetItemString(module_dict, "dtype", (PyObject *)dtype);
    return module;
}
