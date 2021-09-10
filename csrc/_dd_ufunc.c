#include <Python.h>
#include <math.h>
#include <stdio.h>
#include <stdalign.h>

#include "dd_arith.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

/**
 * Allows parameter to be marked unused
 */
#define MARK_UNUSED(x)  do { (void)(x); } while(false)

/* ------------------------ DDouble object ----------------------- */

static PyObject *module = NULL;
static PyObject *numpy_module = NULL;
static int type_num = NPY_CDOUBLE;  //FIXME

static PyTypeObject *pyddouble_type = NULL;

typedef struct {
    PyObject_HEAD;
    ddouble x;
} PyDDouble;

static bool PyDDouble_Check(PyObject* object)
{
    return PyObject_IsInstance(object, (PyObject *)pyddouble_type);
}

static PyObject *PyDDouble_Wrap(ddouble x)
{
    PyDDouble *obj = (PyDDouble *) pyddouble_type->tp_alloc(pyddouble_type, 0);
    if (obj != NULL)
        obj->x = x;
    return (PyObject *)obj;
}

static ddouble PyDDouble_Unwrap(PyObject *arg)
{
    return ((PyDDouble *)arg)->x;
}

static bool PyDDouble_Cast(PyObject *arg, ddouble *out)
{
    if (PyDDouble_Check(arg)) {
        *out = PyDDouble_Unwrap(arg);
    } else if (PyFloat_Check(arg)) {
        double val = PyFloat_AsDouble(arg);
        *out = (ddouble) {val, 0.0};
    } else if (PyLong_Check(arg)) {
        long val = PyLong_AsLong(arg);
        *out = (ddouble) {val, 0.0};
    } else if (PyArray_IsScalar(arg, Float)) {
        float val;
        PyArray_ScalarAsCtype(arg, &val);
        *out = (ddouble) {val, 0.0};
    } else if (PyArray_IsScalar(arg, Double)) {
        double val;
        PyArray_ScalarAsCtype(arg, &val);
        *out = (ddouble) {val, 0.0};
    } else if (PyArray_IsZeroDim(arg)) {
        PyArrayObject* arr = (PyArrayObject *)arg;
        if (PyArray_TYPE(arr) == type_num) {
            *out = *(ddouble *)PyArray_DATA(arr);
        } else {
            arr = (PyArrayObject *)PyArray_Cast(arr, type_num);
            if (!PyErr_Occurred())
                *out = *(ddouble *)PyArray_DATA(arr);
            else
                *out = nanq();
            Py_XDECREF(arr);
        }
    } else {
        *out = nanq();
        PyErr_SetString(PyExc_ValueError, "no conversion");
    }
    return !PyErr_Occurred();
}

PyObject* PyDDouble_New(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *arg = NULL;
    if (PyArg_ParseTuple(args, "O", &arg) < 0)
        return NULL;

    ddouble val;
    if (PyDDouble_Check(arg)) {
        Py_INCREF(arg);
        return arg;
    } else  if (PyDDouble_Cast(arg, &val)) {
        return PyDDouble_Wrap(val);
    } else {
        PyErr_Format(PyExc_TypeError, "expected ddouble, got %s",
                     arg->ob_type->tp_name);
        return NULL;
    }
}

static PyObject* PyDDouble_Float(PyObject* self)
{
    ddouble x = PyDDouble_Unwrap(self);
    return PyFloat_FromDouble(x.hi);
}

static PyObject* PyDDouble_Int(PyObject* self)
{
    ddouble x = PyDDouble_Unwrap(self);
    return PyFloat_FromDouble((long) x.hi);
}

#define PYWRAP_UNARY(name, inner)                                       \
    static PyObject* name(PyObject* _x)                                 \
    {                                                                   \
        ddouble r, x;                                                   \
        x = PyDDouble_Unwrap(_x);                                       \
        r = inner(x);                                                   \
        return PyDDouble_Wrap(r);                                       \
    }

#define PYWRAP_BINARY(name, inner)                                      \
    static PyObject* name(PyObject* _x, PyObject* _y)                   \
    {                                                                   \
        ddouble r, x, y;                                                \
        if (PyDDouble_Cast(_x, &x) && PyDDouble_Cast(_y, &y)) {         \
            r = inner(x, y);                                            \
            return PyDDouble_Wrap(r);                                   \
        } else {                                                        \
            return NULL;                                                \
        }                                                               \
    }

#define PYWRAP_INPLACE(name, inner)                                     \
    static PyObject* name(PyObject* _self, PyObject* _y)                \
    {                                                                   \
        PyDDouble *self = (PyDDouble *)_self;                           \
        ddouble y;                                                      \
        if (PyDDouble_Cast(_y, &y)) {                                   \
            self->x = inner(self->x, y);                                \
            Py_XINCREF(_self);                                          \
            return _self;                                               \
        } else {                                                        \
            return NULL;                                                \
        }                                                               \
    }

PYWRAP_UNARY(PyDDouble_Positive, posq)
PYWRAP_UNARY(PyDDouble_Negative, negq)
PYWRAP_UNARY(PyDDouble_Absolute, absq)

PYWRAP_BINARY(PyDDouble_Add, addqq)
PYWRAP_BINARY(PyDDouble_Subtract, subqq)
PYWRAP_BINARY(PyDDouble_Multiply, mulqq)
PYWRAP_BINARY(PyDDouble_Divide, divqq)

PYWRAP_INPLACE(PyDDouble_InPlaceAdd, addqq)
PYWRAP_INPLACE(PyDDouble_InPlaceSubtract, subqq)
PYWRAP_INPLACE(PyDDouble_InPlaceMultiply, mulqq)
PYWRAP_INPLACE(PyDDouble_InPlaceDivide, divqq)

static int PyDDouble_Nonzero(PyObject* _x)
{
    ddouble x = PyDDouble_Unwrap(_x);
    return !(x.hi == 0);
}

PyObject* PyDDouble_RichCompare(PyObject* _x, PyObject* _y, int op)
{
    ddouble x, y;
    if (!PyDDouble_Cast(_x, &x) || !PyDDouble_Cast(_y, &y))
        return PyGenericArrType_Type.tp_richcompare(_x, _y, op);

    bool result;
    switch (op) {
    case Py_LT:
        result = lessqq(x, y);
        break;
    case Py_LE:
        result = lessequalqq(x, y);
        break;
    case Py_EQ:
        result = equalqq(x, y);
        break;
    case Py_NE:
        result = notequalqq(x, y);
        break;
    case Py_GT:
        result = greaterqq(x, y);
        break;
    case Py_GE:
        result = greaterequalqq(x, y);
        break;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Invalid op type");
        return NULL;
    }
    return PyBool_FromLong(result);
}

Py_hash_t PyDDouble_Hash(PyObject *_x)
{
    ddouble x = PyDDouble_Unwrap(_x);

    int exp;
    double mantissa;
    mantissa = frexp(x.hi, &exp);
    return SSIZE_MAX * mantissa + exp;
}

PyObject *PyDDouble_Str(PyObject *self)
{
    char out[200];
    ddouble x = PyDDouble_Unwrap(self);
    snprintf(out, 200, "%.16g", x.hi);
    return PyUnicode_FromString(out);
}

PyObject *PyDDouble_Repr(PyObject *self)
{
    char out[200];
    ddouble x = PyDDouble_Unwrap(self);
    snprintf(out, 200, "ddouble(%.16g)", x.hi);
    return PyUnicode_FromString(out);
}

int make_ddouble_type()
{
    static PyNumberMethods ddouble_as_number = {
        .nb_add = PyDDouble_Add,
        .nb_subtract = PyDDouble_Subtract,
        .nb_multiply = PyDDouble_Multiply,
        .nb_true_divide = PyDDouble_Divide,
        .nb_inplace_add = PyDDouble_InPlaceAdd,
        .nb_inplace_subtract = PyDDouble_InPlaceSubtract,
        .nb_inplace_multiply = PyDDouble_InPlaceMultiply,
        .nb_inplace_true_divide = PyDDouble_InPlaceDivide,
        .nb_negative = PyDDouble_Negative,
        .nb_positive = PyDDouble_Positive,
        .nb_absolute = PyDDouble_Absolute,
        .nb_bool = PyDDouble_Nonzero,
        .nb_int = PyDDouble_Int,
        .nb_float = PyDDouble_Float,
        };
    static PyTypeObject ddouble_type = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "ddouble",
        .tp_basicsize = sizeof(PyDDouble),
        .tp_repr = PyDDouble_Repr,
        .tp_as_number = &ddouble_as_number,
        .tp_hash = PyDDouble_Hash,
        .tp_str = PyDDouble_Str,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "double-double floating point type",
        .tp_richcompare = PyDDouble_RichCompare,
        .tp_new = PyDDouble_New,
        };

    ddouble_type.tp_base = &PyGenericArrType_Type;
    if (PyType_Ready(&ddouble_type) < 0)
        return -1;

    pyddouble_type = &ddouble_type;
    return PyModule_AddObject(module, "ddouble", (PyObject *)pyddouble_type);
}

/* ------------------------------ Descriptor ----------------------------- */

static PyObject *NPyDDouble_GetItem(void *data, void *arr)
{
    ddouble x = *(ddouble *)data;
    return PyDDouble_Wrap(x);
    MARK_UNUSED(arr);
}

static int NPyDDouble_SetItem(PyObject *item, void *data, void *arr)
{
    ddouble x;
    if (!PyDDouble_Cast(item, &x))
        return -1;
    *(ddouble *)data = x;
    return 0;
    MARK_UNUSED(arr);
}

static int NPyDDouble_Compare(const void *_a, const void *_b, void *arr)
{
    ddouble a = *(const ddouble *)_a;
    ddouble b = *(const ddouble *)_b;

    if (lessqq(a, b))
        return -1;
    if (greaterqq(a, b))
        return 1;
    if (isnanq(b))
        return 1;
    return 0;
    MARK_UNUSED(arr);
}

static void NPyDDouble_CopySwapN(void *_d, npy_intp sd, void *_s, npy_intp ss,
                                 npy_intp ii, int swap, void* arr)
{
    if (_s == NULL)
        return;
    if (swap) {
        for (npy_intp i = 0; i != ii; ++i, _d += sd, _s += ss) {
            ddouble *s = _s, *d = _d, tmp;
            tmp = *d;
            *d = *s;
            *s = tmp;
        }
    } else {
        for (npy_intp i = 0; i != ii; ++i, _d += sd, _s += ss) {
            ddouble *s = _s, *d = _d;
            *d = *s;
        }
    }
    MARK_UNUSED(arr);
}

static void NPyDDouble_CopySwap(void *_d, void *_s, int swap, void* arr)
{
    ddouble *s = _s, *d = _d, tmp;
    if (_s == NULL)
        return;
    if (swap) {
        tmp = *d;
        *d = *s;
        *s = tmp;
    } else {
        *d = *s;
    }
    MARK_UNUSED(arr);
}

static npy_bool NPyDDouble_NonZero(void *data, void *arr)
{
    ddouble x = *(ddouble *)data;
    return !iszeroq(x);
    MARK_UNUSED(arr);
}

static int NPyDDouble_Fill(void *_buffer, npy_intp ii, void *arr)
{
    // Fill with linear array
    ddouble *buffer = (ddouble *)_buffer;
    if (ii < 2)
        return -1;

    ddouble curr = buffer[1];
    ddouble step = subqq(curr, buffer[0]);
    for (npy_intp i = 2; i != ii; ++i) {
        curr = addqq(curr, step);
        buffer[i] = curr;
    }
    return 0;
    MARK_UNUSED(arr);
}

static int NPyDDouble_FillWithScalar(void *_buffer, npy_intp ii, void *_value,
                                      void *arr)
{
    ddouble *buffer = (ddouble *)_buffer;
    ddouble value = *(ddouble *)_value;
    for (npy_intp i = 0; i < ii; ++i)
        buffer[i] = value;
    return 0;
    MARK_UNUSED(arr);
}

static void NPyDDouble_DotFunc(void *_in1, npy_intp is1, void *_in2,
                               npy_intp is2, void *_out, npy_intp ii, void *arr)
{
    ddouble out = Q_ZERO;
    for (npy_intp i = 0; i < ii; ++i, _in1 += is1, _in2 += is2) {
        ddouble in1 = *(ddouble *)_in1, in2 = *(ddouble *)_in2;
        out = addqq(out, mulqq(in1, in2));
    }
    *(ddouble *)_out = out;
    MARK_UNUSED(arr);
}

static int NPyDDouble_ArgMax(void *_data, npy_intp n, npy_intp *max_ind,
                             void *arr)
{
    ddouble *data = (ddouble *)_data;
    ddouble max_val = negq(infq());
    for (npy_intp i = 0; i < n; ++i) {
        if (greaterqq(data[i], max_val)) {
            max_val = data[i];
            *max_ind = i;
        }
    }
    return 0;
    MARK_UNUSED(arr);
}

static int NPyDDouble_ArgMin(void *_data, npy_intp n, npy_intp *min_ind,
                             void *arr)
{
    ddouble *data = (ddouble *)_data;
    ddouble min_val = infq();
    for (npy_intp i = 0; i < n; ++i) {
        if (lessqq(data[i], min_val)) {
            min_val = data[i];
            *min_ind = i;
        }
    }
    return 0;
    MARK_UNUSED(arr);
}

static int make_dtype()
{
    /* Check if another module has registered a ddouble type.
     */
    type_num = PyArray_TypeNumFromName("ddouble");
    if (type_num != NPY_NOTYPE) {
        return type_num;
    }

    static PyArray_ArrFuncs ddouble_arrfuncs;
    static PyArray_Descr ddouble_dtype = {
        PyObject_HEAD_INIT(NULL)

        /* We must register bfloat16 with a kind other than "f", because numpy
         * considers two types with the same kind and size to be equal, but
         * float128 != ddouble.  The downside of this is that NumPy scalar
         * promotion does not work with ddoubles.
         */
        .kind = 'V',
        .type = 'X',
        .byteorder = '=',

        /* TODO: not sure why this must be there */
        .flags = NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM,
        .elsize = sizeof(ddouble),
        .alignment = alignof(ddouble),
        .hash = -1
        };

    ddouble_dtype.typeobj = pyddouble_type;
    ddouble_dtype.f = &ddouble_arrfuncs;
    Py_TYPE(&ddouble_dtype) = &PyArrayDescr_Type;

    PyArray_InitArrFuncs(&ddouble_arrfuncs);
    ddouble_arrfuncs.getitem = NPyDDouble_GetItem;
    ddouble_arrfuncs.setitem = NPyDDouble_SetItem;
    ddouble_arrfuncs.compare = NPyDDouble_Compare;
    ddouble_arrfuncs.copyswapn = NPyDDouble_CopySwapN;
    ddouble_arrfuncs.copyswap = NPyDDouble_CopySwap;
    ddouble_arrfuncs.nonzero = NPyDDouble_NonZero;
    ddouble_arrfuncs.fill = NPyDDouble_Fill;
    ddouble_arrfuncs.fillwithscalar = NPyDDouble_FillWithScalar;
    ddouble_arrfuncs.dotfunc = NPyDDouble_DotFunc;
    ddouble_arrfuncs.argmax = NPyDDouble_ArgMax;
    ddouble_arrfuncs.argmin = NPyDDouble_ArgMin;

    type_num = PyArray_RegisterDataType(&ddouble_dtype);
    return type_num;
}

/* ------------------------------- Ufuncs ----------------------------- */


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
ULOOP_BINARY(u_copysignqq, copysignqq, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_copysignqd, copysignqd, ddouble, ddouble, double)
ULOOP_BINARY(u_copysigndq, copysigndq, ddouble, double, ddouble)
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
ULOOP_BINARY(u_hypotqq, hypotqq, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_hypotdq, hypotdq, ddouble, double, ddouble)
ULOOP_BINARY(u_hypotqd, hypotqd, ddouble, ddouble, double)

ULOOP_UNARY(u_signbitq, signbitq, bool, ddouble)
ULOOP_UNARY(u_signq, signq, ddouble, ddouble)
ULOOP_UNARY(u_isfiniteq, isfiniteq, bool, ddouble)
ULOOP_UNARY(u_isinfq, isinfq, bool, ddouble)
ULOOP_UNARY(u_isnanq, isnanq, bool, ddouble)
ULOOP_UNARY(u_negq, negq, ddouble, ddouble)
ULOOP_UNARY(u_posq, posq, ddouble, ddouble)
ULOOP_UNARY(u_absq, absq, ddouble, ddouble)
ULOOP_UNARY(u_reciprocalq, reciprocalq, ddouble, ddouble)
ULOOP_UNARY(u_sqrq, sqrq, ddouble, ddouble)
ULOOP_UNARY(u_roundq, roundq, ddouble, ddouble)
ULOOP_UNARY(u_floorq, floorq, ddouble, ddouble)
ULOOP_UNARY(u_ceilq, ceilq, ddouble, ddouble)
ULOOP_UNARY(u_iszeroq, iszeroq, bool, ddouble)
ULOOP_UNARY(u_isoneq, isoneq, bool, ddouble)
ULOOP_UNARY(u_ispositiveq, ispositiveq, bool, ddouble)
ULOOP_UNARY(u_isnegativeq, isnegativeq, bool, ddouble)
ULOOP_UNARY(u_sqrtq, sqrtq, ddouble, ddouble)
ULOOP_UNARY(u_expq, expq, ddouble, ddouble)
ULOOP_UNARY(u_expm1q, expm1q, ddouble, ddouble)
ULOOP_UNARY(u_logq, logq, ddouble, ddouble)
ULOOP_UNARY(u_sinq, sinq, ddouble, ddouble)
ULOOP_UNARY(u_cosq, cosq, ddouble, ddouble)
ULOOP_UNARY(u_sinhq, sinhq, ddouble, ddouble)
ULOOP_UNARY(u_coshq, coshq, ddouble, ddouble)
ULOOP_UNARY(u_tanhq, tanhq, ddouble, ddouble)


static int register_binary(PyUFuncGenericFunction dq_func,
        PyUFuncGenericFunction qd_func, PyUFuncGenericFunction qq_func,
        int ret_dtype, const char *name)
{
    PyUFuncObject *ufunc;
    int *arg_types = NULL, retcode = 0;

    ufunc = (PyUFuncObject *)PyObject_GetAttrString(numpy_module, name);
    if (ufunc == NULL) goto error;

    arg_types = PyMem_New(int, 3 * 3);
    if (arg_types == NULL) goto error;

    arg_types[0] = NPY_DOUBLE;
    arg_types[1] = type_num;
    arg_types[2] = ret_dtype;
    retcode = PyUFunc_RegisterLoopForType(ufunc, type_num,
                                          dq_func, arg_types, NULL);
    if (retcode < 0) goto error;

    arg_types[3] = type_num;
    arg_types[4] = NPY_DOUBLE;
    arg_types[5] = ret_dtype;
    retcode = PyUFunc_RegisterLoopForType(ufunc, type_num,
                                          dq_func, arg_types + 3, NULL);
    if (retcode < 0) goto error;

    arg_types[6] = type_num;
    arg_types[7] = type_num;
    arg_types[8] = ret_dtype;
    retcode = PyUFunc_RegisterLoopForType(ufunc, type_num,
                                          dq_func, arg_types + 6, NULL);
    if (retcode < 0) goto error;
    return 0;

error:
    return -1;
}

static int register_unary(PyUFuncGenericFunction func, int ret_dtype,
                          const char *name)
{
    PyUFuncObject *ufunc;
    int *arg_types = NULL, retcode = 0;

    ufunc = (PyUFuncObject *)PyObject_GetAttrString(numpy_module, name);
    if (ufunc == NULL) goto error;

    arg_types = PyMem_New(int, 2);
    if (arg_types == NULL) goto error;

    arg_types[0] = type_num;
    arg_types[1] = ret_dtype;
    retcode = PyUFunc_RegisterLoopForType(ufunc, type_num,
                                          func, arg_types, NULL);
    if (retcode < 0) goto error;
    return 0;

error:
    return -1;
}

static int register_ufuncs()
{
    register_unary(u_negq, type_num, "negative");
    register_unary(u_posq, type_num, "positive");
    register_unary(u_absq, type_num, "absolute");
    register_unary(u_reciprocalq, type_num, "reciprocal");
    register_unary(u_sqrq, type_num, "square");
    register_unary(u_sqrtq, type_num, "sqrt");
    register_unary(u_signbitq, NPY_BOOL, "signbit");
    register_unary(u_isfiniteq, NPY_BOOL, "isfinite");
    register_unary(u_isinfq, NPY_BOOL, "isinf");
    register_unary(u_isnanq, NPY_BOOL, "isnan");

    register_unary(u_roundq, type_num, "rint");
    register_unary(u_floorq, type_num, "floor");
    register_unary(u_ceilq, type_num, "ceil");
    register_unary(u_expq, type_num, "exp");
    register_unary(u_expm1q, type_num, "expm1");
    register_unary(u_logq, type_num, "log");
    register_unary(u_sinq, type_num, "sin");
    register_unary(u_cosq, type_num, "cos");
    register_unary(u_sinhq, type_num, "sinh");
    register_unary(u_coshq, type_num, "cosh");
    register_unary(u_tanhq, type_num, "tanh");

    register_unary(u_iszeroq, NPY_BOOL, "iszero");
    register_unary(u_isoneq, NPY_BOOL, "isone");
    register_unary(u_ispositiveq, NPY_BOOL, "ispositive");
    register_unary(u_isnegativeq, NPY_BOOL, "isnegative");
    register_unary(u_signq, type_num, "sign");

    register_binary(u_adddq, u_addqd, u_addqq, type_num, "add");
    register_binary(u_subdq, u_subqd, u_subqq, type_num, "subtract");
    register_binary(u_muldq, u_mulqd, u_mulqq, type_num, "multiply");
    register_binary(u_divdq, u_divqd, u_divqq, type_num, "true_divide");

    register_binary(u_equaldq, u_equalqd, u_equalqq, NPY_BOOL, "equal");
    register_binary(u_notequaldq, u_notequalqd, u_notequalqq, NPY_BOOL,
                    "not_equal");
    register_binary(u_greaterdq, u_greaterqd, u_greaterqq, NPY_BOOL, "greater");
    register_binary(u_lessdq, u_lessqd, u_lessqq, NPY_BOOL, "less");
    register_binary(u_greaterequaldq, u_greaterequalqd, u_greaterequalqq,
                    NPY_BOOL, "greater_equal");
    register_binary(u_lessequaldq, u_lessequalqd, u_lessequalqq, NPY_BOOL,
                    "less_equal");
    register_binary(u_fmindq, u_fminqd, u_fminqq, type_num, "fmin");
    register_binary(u_fmaxdq, u_fmaxqd, u_fmaxqq, type_num, "fmax");

    register_binary(u_copysigndq, u_copysignqd, u_copysignqq, type_num,
                    "copysign");
    register_binary(u_hypotdq, u_hypotqd, u_hypotqq, type_num, "hypot");
    return 0;
}

int register_dtype_in_dicts()
{
    PyObject *type_dict = NULL;

    type_dict = PyObject_GetAttrString(numpy_module, "sctypeDict");
    if (type_dict == NULL) goto error;

    if (PyDict_SetItemString(type_dict, "ddouble",
                             (PyObject *)pyddouble_type) < 0)
        goto error;
    return 0;

error:
    Py_XDECREF(type_dict);
    return -1;
}

/* ----------------------- Python stuff -------------------------- */

static const char DDOUBLE_WRAP = NPY_CDOUBLE;

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

    if (make_ddouble_type() < 0)
        return NULL;
    if (make_dtype() < 0)
        return NULL;

    numpy_module = PyImport_ImportModule("numpy");
    if (numpy_module == NULL)
        return NULL;

    PyArray_Descr *fancy_dtype = PyArray_DescrFromType(type_num);
    PyModule_AddObject(module, "fancy_dtype", (PyObject *)fancy_dtype);

    register_ufuncs();
    if (register_dtype_in_dicts() < 0)
        return NULL;

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
