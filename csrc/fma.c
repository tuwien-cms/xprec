#include "Python.h"
#include "math.h"

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

static void double_fma(char **args, npy_intp *dimensions,
                       npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];

    char *in1 = args[0], *in2 = args[1], *in3 = args[2], *out1 = args[3];
    npy_intp istep1 = steps[0], istep2 = steps[1], istep3 = steps[2],
             ostep1 = steps[3];

    for (i = 0; i < n; i++) {
        *(double *)out1 = fma(*(double *)in1, *(double *)in2, *(double *)in3);

        in1 += istep1;
        in2 += istep2;
        in3 += istep3;
        out1 += ostep1;
    }
}

static PyUFuncGenericFunction funcs[] = {&double_fma};
static char types[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
static void *data[] = {NULL};


PyMODINIT_FUNC PyInit__fma(void)
{
    PyObject *module, *ufunc, *dict;
    module = PyModule_Create(&module_def);
    if (!module)
        return NULL;

    import_array();
    import_umath();

    ufunc = PyUFunc_FromFuncAndData(funcs, data, types, 1, 3, 1,
                                    PyUFunc_None, "fma", "FMA docstring", 0);

    dict = PyModule_GetDict(module);
    PyDict_SetItemString(dict, "fma", ufunc);
    Py_DECREF(ufunc);

    return module;
}
