#include "Python.h"
#include "math.h"
#include "stdio.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

// Module definition

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


// Caution: loop definitions must precede PyMODINIT_FUNC

static void double_fma(char **args, const npy_intp *dimensions,
                       const npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1], *in3 = args[2], *out1 = args[3];
    npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], os1 = steps[3];

    for (i = 0; i < n; i++) {
        *(double *)out1 = fma(*(double *)in1, *(double *)in2, *(double *)in3);

        in1 += is1;
        in2 += is2;
        in3 += is3;
        out1 += os1;
    }
}

static PyUFuncGenericFunction fma_funcs[] = {&double_fma};
static char fma_types[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
static void *fma_data[] = {NULL};


struct ddouble {
    double x;
    double e;
};


static void add_ddouble(char **args, const npy_intp *dimensions,
                        const npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *_in1 = args[0], *_in2 = args[1], *_out1 = args[2];
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];

    for (i = 0; i < n; i++) {
        // const struct ddouble *lhs = (const struct ddouble *)_in1;
        // const struct ddouble *rhs = (const struct ddouble *)_in2;
        // struct ddouble *out = (struct ddouble *)_out1;

        // out->x = lhs->x + rhs->x;
        // out->x = lhs->e - rhs->e;
        const double *lhs = (const double *)_in1;
        const double *rhs = (const double *)_in2;
        double *out = (double *)_out1;

        out[0] = lhs[0] + rhs[1];
        out[1] = lhs[1] - rhs[1];

        _in1 += is1;
        _in2 += is2;
        _out1 += os1;
    }
}

#define DDOUBLE_BINARY(name, loop_func) 1



// Init routine

PyMODINIT_FUNC PyInit__fma(void)
{
    PyObject *module, *dict;
    PyObject *fma_ufunc, *add_dd_ufunc;

    PyObject *dtype_tuple;
    PyArray_Descr *dtype;
    PyArray_Descr *dtypes[3];

    // Create module
    module = PyModule_Create(&module_def);
    if (!module)
        return NULL;

    // Initialize numpy things
    import_array();
    import_umath();

    // Create ufuncs
    fma_ufunc = PyUFunc_FromFuncAndData(
            fma_funcs, fma_data, fma_types, 1, 3, 1, PyUFunc_None, "fma",
            "FMA docstring", 0);

    // Build dtypes
    dtype_tuple = Py_BuildValue("[(s, s), (s, s)]", "x", "d", "e", "d");
    PyArray_DescrConverter(dtype_tuple, &dtype);
    Py_DECREF(dtype_tuple);

    dtypes[0] = dtype;
    dtypes[1] = dtype;
    dtypes[2] = dtype;

    // Create dummy ufuncs
    add_dd_ufunc = PyUFunc_FromFuncAndData(
            NULL, NULL, NULL, 0, 2, 1, PyUFunc_None, "add_dd",
            "add_dd docstring", 0);

    // Register for dtype
    PyUFunc_RegisterLoopForDescr(
            (PyUFuncObject *)add_dd_ufunc, dtype, &add_ddouble, dtypes, NULL);

    // Register ufuncs
    dict = PyModule_GetDict(module);
    PyDict_SetItemString(dict, "fma", fma_ufunc);
    PyDict_SetItemString(dict, "add_dd", add_dd_ufunc);
    Py_DECREF(fma_ufunc);


    return module;
}
