#include "Python.h"
#include "math.h"
#include "stdio.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"



/**
 * Type for double-double calculations
 */
struct ddouble {
    double x;
    double e;
};

/**
 * Create ufunc loop routine for a unary operation
 */
#define DDOUBLE_UNARY_FUNCTION(func_name, inner_func)                   \
    static void func_name(char **args, const npy_intp *dimensions,      \
                          const npy_intp* steps, void* data)            \
    {                                                                   \
        assert (sizeof(struct ddouble) == 2 * sizeof(double));          \
        npy_intp i;                                                     \
        npy_intp n = dimensions[0];                                     \
        char *_in1 = args[0], *_out1 = args[1];                         \
        npy_intp is1 = steps[0], os1 = steps[1];                        \
                                                                        \
        for (i = 0; i < n; i++) {                                       \
            const struct ddouble *in = (const struct ddouble *)_in1;    \
            struct ddouble *out = (struct ddouble *)_out1;              \
            *out = func(*in);                                           \
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
        assert (sizeof(struct ddouble) == 2 * sizeof(double));          \
        npy_intp i;                                                     \
        npy_intp n = dimensions[0];                                     \
        char *_in1 = args[0], *_in2 = args[1], *_out1 = args[2];        \
        npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];        \
                                                                        \
        for (i = 0; i < n; i++) {                                       \
            const struct ddouble *lhs = (const struct ddouble *)_in1;   \
            const struct ddouble *rhs = (const struct ddouble *)_in2;   \
            struct ddouble *out = (struct ddouble *)_out1;              \
            *out = inner_func(*lhs, *rhs);                              \
                                                                        \
            _in1 += is1;                                                \
            _in2 += is2;                                                \
            _out1 += os1;                                               \
        }                                                               \
    }


inline struct ddouble mytest(struct ddouble lhs, struct ddouble rhs)
{
    struct ddouble result;
    result.x = lhs.x + rhs.x;
    result.e = lhs.e - rhs.e;
    return result;
}
DDOUBLE_BINARY_FUNCTION(add_ddouble, mytest)




static PyArray_Descr *make_ddouble_dtype()
{
    PyObject *dtype_tuple;
    PyArray_Descr *dtype;

    dtype_tuple = Py_BuildValue("[(s, s), (s, s)]", "x", "d", "e", "d");
    PyArray_DescrConverter(dtype_tuple, &dtype);
    Py_DECREF(dtype_tuple);
    return dtype;
}

static void ddouble_binary(PyArray_Descr *dtype, PyObject *module_dict,
                           PyUFuncGenericFunction func, const char *name,
                           const char *docstring)
{
    PyObject *ufunc;
    PyArray_Descr *dtypes[] = {dtype, dtype, dtype};

    ufunc = PyUFunc_FromFuncAndData(
                NULL, NULL, NULL, 0, 2, 1, PyUFunc_None, name, docstring, 0);
    PyUFunc_RegisterLoopForDescr(
                (PyUFuncObject *)ufunc, dtype, func, dtypes, NULL);
    PyDict_SetItemString(module_dict, name, ufunc);
    Py_DECREF(ufunc);
}

// Init routine
PyMODINIT_FUNC PyInit__fma(void)
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
    dtype = make_ddouble_dtype();

    /* Create ufuncs */
    ddouble_binary(dtype, module_dict, add_ddouble, "add_dd", "docstring");

    return module;
}
