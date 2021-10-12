/* Python extension module for the ddouble data type.
 *
 * Code is adapted from tensorflow's bfloat16 extension type, found here:
 * `tensorflow/python/lib/core/bfloat16.cc` and licensed Apache 2.0.
 *
 * Copyright (C) 2021 Markus Wallerberger and others
 * SPDX-License-Identifier: MIT
 */
#include <Python.h>
#include <structmember.h>

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

#ifdef _MSC_VER
#define alignof __alignof
#endif

/* ------------------------ DDouble object ----------------------- */

static PyObject *module = NULL;
static PyObject *numpy_module = NULL;
static int type_num = -1;  //FIXME

static PyTypeObject *pyddouble_type = NULL;
static PyObject *pyddouble_finfo = NULL;

typedef struct {
    PyObject_HEAD
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
        PyErr_Format(PyExc_TypeError,
            "Cannot cast instance of %s to ddouble scalar",
            arg->ob_type->tp_name);
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

#define PYWRAP_BINARY(name, inner, tp_inner_op)                         \
    static PyObject* name(PyObject* _x, PyObject* _y)                   \
    {                                                                   \
        ddouble r, x, y;                                                \
        if (PyArray_Check(_y))                                          \
            return PyArray_Type.tp_as_number->tp_inner_op(_x, _y);      \
        if (PyDDouble_Cast(_x, &x) && PyDDouble_Cast(_y, &y)) {         \
            r = inner(x, y);                                            \
            return PyDDouble_Wrap(r);                                   \
        }                                                               \
        return NULL;                                                    \
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

PYWRAP_BINARY(PyDDouble_Add, addqq, nb_add)
PYWRAP_BINARY(PyDDouble_Subtract, subqq, nb_subtract)
PYWRAP_BINARY(PyDDouble_Multiply, mulqq, nb_multiply)
PYWRAP_BINARY(PyDDouble_Divide, divqq, nb_true_divide)

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
    return (Py_hash_t)(LONG_MAX * mantissa) + exp;
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
    snprintf(out, 200, "ddouble(%.16g+%.16g)", x.hi, x.lo);
    return PyUnicode_FromString(out);
}

PyObject *PyDDoubleGetFinfo(PyObject *self, PyObject *_dummy)
{
    Py_INCREF(pyddouble_finfo);
    return pyddouble_finfo;
    MARK_UNUSED(_dummy);
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
    static PyMethodDef ddouble_methods[] = {
        {"__finfo__", PyDDoubleGetFinfo, METH_NOARGS | METH_CLASS,
         "floating point information for type"},
        {NULL}
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
        .tp_methods = ddouble_methods
        };

    ddouble_type.tp_base = &PyFloatingArrType_Type;
    if (PyType_Ready(&ddouble_type) < 0)
        return -1;

    pyddouble_type = &ddouble_type;
    return PyModule_AddObject(module, "ddouble", (PyObject *)pyddouble_type);
}

/* --------------------- Ddouble Finfo object -------------------- */

typedef struct {
    PyObject_HEAD
    PyObject *dtype;    // which dtype
    int bits;           // number of bits
    PyObject *max;      // largest positive number
    PyObject *min;      // largest negative number
    PyObject *eps;      // machine epsilon (spacing)
    int nexp;           // number of exponent bits
    int nmant;          // number of mantissa bits
    PyObject *machar;   // machar object (unused)
} PyDDoubleFInfo;

static PyTypeObject *PyDDoubleFinfoType;

static PyObject *PPyDDoubleFInfo_Make()
{
    PyDDoubleFInfo *self =
        (PyDDoubleFInfo *) PyDDoubleFinfoType->tp_alloc(PyDDoubleFinfoType, 0);
    if (self == NULL)
        return NULL;

    Py_INCREF(Py_None);
    self->dtype = (PyObject *)PyArray_DescrFromType(type_num);
    self->bits = CHAR_BIT * sizeof(ddouble);
    self->max = PyDDouble_Wrap(Q_MAX);
    self->min = PyDDouble_Wrap(Q_MIN);
    self->eps = PyDDouble_Wrap(Q_EPS);
    self->nexp = 11;
    self->nmant = 104;
    self->machar = Py_None;
    return (PyObject *)self;
}

static int make_finfo()
{
    static PyMemberDef finfo_members[] = {
        {"dtype",  T_OBJECT_EX, offsetof(PyDDoubleFInfo, dtype),  READONLY},
        {"bits",   T_INT,       offsetof(PyDDoubleFInfo, bits),   READONLY},
        {"max",    T_OBJECT_EX, offsetof(PyDDoubleFInfo, max),    READONLY},
        {"min",    T_OBJECT_EX, offsetof(PyDDoubleFInfo, min),    READONLY},
        {"eps",    T_OBJECT_EX, offsetof(PyDDoubleFInfo, eps),    READONLY},
        {"nexp",   T_INT,       offsetof(PyDDoubleFInfo, nexp),   READONLY},
        {"nmant",  T_INT,       offsetof(PyDDoubleFInfo, nmant),  READONLY},
        {"machar", T_OBJECT_EX, offsetof(PyDDoubleFInfo, machar), READONLY},
        {NULL}
        };
    static PyTypeObject finfo_type = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "ddouble_finfo",
        .tp_basicsize = sizeof(PyDDoubleFInfo),
        .tp_members = finfo_members,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "finfo type"
        };

    if (PyType_Ready(&finfo_type) < 0)
        return -1;

    PyDDoubleFinfoType = &finfo_type;
    pyddouble_finfo = PPyDDoubleFInfo_Make();
    if (pyddouble_finfo == NULL)
        return -1;

    return 0;
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
    char *_cd = (char *)_d, *_cs = (char *)_s;
    if (swap) {
        for (npy_intp i = 0; i != ii; ++i, _cd += sd, _cs += ss) {
            ddouble *s = (ddouble *)_cs, *d = (ddouble *)_cd, tmp;
            tmp = *d;
            *d = *s;
            *s = tmp;
        }
    } else {
        for (npy_intp i = 0; i != ii; ++i, _cd += sd, _cs += ss) {
            ddouble *s = (ddouble *)_cs, *d = (ddouble *)_cd;
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
    char *_cin1 = (char *)_in1, *_cin2 = (char *)_in2;
    for (npy_intp i = 0; i < ii; ++i, _cin1 += is1, _cin2 += is2) {
        ddouble in1 = *(ddouble *)_cin1, in2 = *(ddouble *)_cin2;
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

        /* We must register ddouble with a kind other than "f", because numpy
         * considers two types with the same kind and size to be equal, but
         * float128 != ddouble.  The downside of this is that NumPy scalar
         * promotion does not work with ddoubles.
         */
        .kind = 'V',
        .type = 'E',
        .byteorder = '=',

        /* NPY_USE_GETITEM is not needed, since we inherit from numpy scalar,
         * which according to the docs means that "standard conversion" is
         * used.  However, we still need to define and register getitem()
         * below, otherwise PyArray_RegisterDataType complains.
         */
        .flags = 0,
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

/* ------------------------------- Casts ------------------------------ */

#define NPY_CAST_FROM(func, from_type)                               \
    static void func(void *_from, void *_to, npy_intp n,             \
                     void *_arr_from, void *_arr_to)                 \
    {                                                                \
        ddouble *to = (ddouble *)_to;                                \
        const from_type *from = (const from_type *)_from;            \
        for (npy_intp i = 0; i < n; ++i)                             \
            to[i] = (ddouble) { from[i], 0.0 };                      \
        MARK_UNUSED(_arr_from);                                      \
        MARK_UNUSED(_arr_to);                                        \
    }

#define NPY_CAST_FROM_I64(func, from_type)                           \
    static void func(void *_from, void *_to, npy_intp n,             \
                     void *_arr_from, void *_arr_to)                 \
    {                                                                \
        static const from_type SPLIT = (from_type)(1) << 32;         \
        ddouble *to = (ddouble *)_to;                                \
        const from_type *from = (const from_type *)_from;            \
        for (npy_intp i = 0; i < n; ++i) {                           \
            from_type lo = from[i] % SPLIT;                          \
            from_type hi = from[i] - lo;                             \
            to[i] = two_sum(hi, lo);                                 \
        }                                                            \
        MARK_UNUSED(_arr_from);                                      \
        MARK_UNUSED(_arr_to);                                        \
    }

#define NPY_CAST_TO(func, to_type)                                   \
    static void func(void *_from, void *_to, npy_intp n,             \
                     void *_arr_from, void *_arr_to)                 \
    {                                                                \
        to_type *to = (to_type *)_to;                                \
        const ddouble *from = (const ddouble *)_from;                \
        for (npy_intp i = 0; i < n; ++i)                             \
            to[i] = (to_type) from[i].hi;                            \
        MARK_UNUSED(_arr_from);                                      \
        MARK_UNUSED(_arr_to);                                        \
    }

// These casts are all loss-less
NPY_CAST_FROM(from_double, double)
NPY_CAST_FROM(from_float, float)
NPY_CAST_FROM(from_bool, bool)
NPY_CAST_FROM(from_int8, int8_t)
NPY_CAST_FROM(from_int16, int16_t)
NPY_CAST_FROM(from_int32, int32_t)
NPY_CAST_FROM(from_uint8, uint8_t)
NPY_CAST_FROM(from_uint16, uint16_t)
NPY_CAST_FROM(from_uint32, uint32_t)

// These casts are also lossless, because we have now 2*54 bits of mantissa
NPY_CAST_FROM_I64(from_int64, int64_t)
NPY_CAST_FROM_I64(from_uint64, uint64_t)

// These casts are all lossy
NPY_CAST_TO(to_double, double)
NPY_CAST_TO(to_float, float)
NPY_CAST_TO(to_bool, bool)
NPY_CAST_TO(to_int8, int8_t)
NPY_CAST_TO(to_int16, int16_t)
NPY_CAST_TO(to_int32, int32_t)
NPY_CAST_TO(to_int64, int64_t)
NPY_CAST_TO(to_uint8, uint8_t)
NPY_CAST_TO(to_uint16, uint16_t)
NPY_CAST_TO(to_uint32, uint32_t)
NPY_CAST_TO(to_uint64, uint64_t)


static bool register_cast(int other_type, PyArray_VectorUnaryFunc from_other,
                         PyArray_VectorUnaryFunc to_other)
{
    PyArray_Descr *other_descr = NULL, *ddouble_descr = NULL;
    int ret;

    other_descr = PyArray_DescrFromType(other_type);
    if (other_descr == NULL) goto error;

    ddouble_descr = PyArray_DescrFromType(type_num);
    if (ddouble_descr == NULL) goto error;

    ret = PyArray_RegisterCastFunc(other_descr, type_num, from_other);
    if (ret < 0) goto error;

    // NPY_NOSCALAR apparently implies that casting is safe?
    ret = PyArray_RegisterCanCast(other_descr, type_num, NPY_NOSCALAR);
    if (ret < 0) goto error;

    ret = PyArray_RegisterCastFunc(ddouble_descr, other_type, to_other);
    if (ret < 0) goto error;
    return true;

error:
    return false;
}

static int register_casts()
{
    bool ok = register_cast(NPY_DOUBLE, from_double, to_double)
        && register_cast(NPY_FLOAT,  from_float,  to_float)
        && register_cast(NPY_BOOL,   from_bool,   to_bool)
        && register_cast(NPY_INT8,   from_int8,   to_int8)
        && register_cast(NPY_INT16,  from_int16,  to_int16)
        && register_cast(NPY_INT32,  from_int32,  to_int32)
        && register_cast(NPY_INT64,  from_int64,  to_int64)
        && register_cast(NPY_UINT8,  from_uint8,  to_uint8)
        && register_cast(NPY_UINT16, from_uint16, to_uint16)
        && register_cast(NPY_UINT32, from_uint32, to_uint32)
        && register_cast(NPY_UINT64, from_uint64, to_uint64);
    return ok ? 0 : -1;
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
        for (npy_intp i = 0; i < n; ++i) {                              \
            out[i * os] = inner_func(a[i * as], b[i * bs]);             \
        }                                                               \
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
ULOOP_UNARY(u_sqrtq, sqrtq, ddouble, ddouble)
ULOOP_UNARY(u_expq, expq, ddouble, ddouble)
ULOOP_UNARY(u_expm1q, expm1q, ddouble, ddouble)
ULOOP_UNARY(u_logq, logq, ddouble, ddouble)
ULOOP_UNARY(u_sinq, sinq, ddouble, ddouble)
ULOOP_UNARY(u_cosq, cosq, ddouble, ddouble)
ULOOP_UNARY(u_sinhq, sinhq, ddouble, ddouble)
ULOOP_UNARY(u_coshq, coshq, ddouble, ddouble)
ULOOP_UNARY(u_tanhq, tanhq, ddouble, ddouble)

static bool register_binary(PyUFuncGenericFunction dq_func,
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
                                          qd_func, arg_types + 3, NULL);
    if (retcode < 0) goto error;

    arg_types[6] = type_num;
    arg_types[7] = type_num;
    arg_types[8] = ret_dtype;
    retcode = PyUFunc_RegisterLoopForType(ufunc, type_num,
                                          qq_func, arg_types + 6, NULL);
    if (retcode < 0) goto error;
    return true;

error:
    return false;
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
    return true;

error:
    return false;
}

static int register_ufuncs()
{
    bool ok = register_unary(u_negq, type_num, "negative")
        && register_unary(u_posq, type_num, "positive")
        && register_unary(u_absq, type_num, "absolute")
        && register_unary(u_reciprocalq, type_num, "reciprocal")
        && register_unary(u_sqrq, type_num, "square")
        && register_unary(u_sqrtq, type_num, "sqrt")
        && register_unary(u_signbitq, NPY_BOOL, "signbit")
        && register_unary(u_isfiniteq, NPY_BOOL, "isfinite")
        && register_unary(u_isinfq, NPY_BOOL, "isinf")
        && register_unary(u_isnanq, NPY_BOOL, "isnan")
        && register_unary(u_roundq, type_num, "rint")
        && register_unary(u_floorq, type_num, "floor")
        && register_unary(u_ceilq, type_num, "ceil")
        && register_unary(u_expq, type_num, "exp")
        && register_unary(u_expm1q, type_num, "expm1")
        && register_unary(u_logq, type_num, "log")
        && register_unary(u_sinq, type_num, "sin")
        && register_unary(u_cosq, type_num, "cos")
        && register_unary(u_sinhq, type_num, "sinh")
        && register_unary(u_coshq, type_num, "cosh")
        && register_unary(u_tanhq, type_num, "tanh")
        && register_unary(u_signq, type_num, "sign")
        && register_binary(u_adddq, u_addqd, u_addqq, type_num, "add")
        && register_binary(u_subdq, u_subqd, u_subqq, type_num, "subtract")
        && register_binary(u_muldq, u_mulqd, u_mulqq, type_num, "multiply")
        && register_binary(u_divdq, u_divqd, u_divqq, type_num, "true_divide")
        && register_binary(u_equaldq, u_equalqd, u_equalqq, NPY_BOOL, "equal")
        && register_binary(u_notequaldq, u_notequalqd, u_notequalqq, NPY_BOOL,
                           "not_equal")
        && register_binary(u_greaterdq, u_greaterqd, u_greaterqq, NPY_BOOL, "greater")
        && register_binary(u_lessdq, u_lessqd, u_lessqq, NPY_BOOL, "less")
        && register_binary(u_greaterequaldq, u_greaterequalqd, u_greaterequalqq,
                           NPY_BOOL, "greater_equal")
        && register_binary(u_lessequaldq, u_lessequalqd, u_lessequalqq, NPY_BOOL,
                           "less_equal")
        && register_binary(u_fmindq, u_fminqd, u_fminqq, type_num, "fmin")
        && register_binary(u_fmaxdq, u_fmaxqd, u_fmaxqq, type_num, "fmax")
        && register_binary(u_fmindq, u_fminqd, u_fminqq, type_num, "minimum")
        && register_binary(u_fmaxdq, u_fmaxqd, u_fmaxqq, type_num, "maximum")
        && register_binary(u_copysigndq, u_copysignqd, u_copysignqq, type_num,
                           "copysign")
        && register_binary(u_hypotdq, u_hypotqd, u_hypotqq, type_num, "hypot");
    return ok ? 0 : -1;
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

static bool constant(ddouble value, const char *name)
{
    // Note that data must be allocated using malloc, not python allocators!
    ddouble *data = malloc(sizeof value);
    *data = value;

    PyArrayObject *array = (PyArrayObject *)
            PyArray_SimpleNewFromData(0, NULL, type_num, data);
    if (array == NULL) return false;

    PyArray_ENABLEFLAGS(array, NPY_ARRAY_OWNDATA);
    PyArray_CLEARFLAGS(array, NPY_ARRAY_WRITEABLE);

    PyModule_AddObject(module, name, (PyObject *)array);
    return true;
}

static int register_constants()
{
    bool ok = constant(Q_MAX, "MAX")
        && constant(Q_MIN, "MIN")
        && constant(Q_EPS, "EPS")
        && constant(Q_2PI, "TWOPI")
        && constant(Q_PI, "PI")
        && constant(Q_PI_2, "PI_2")
        && constant(Q_PI_4, "PI_4")
        && constant(Q_E, "E")
        && constant(Q_LOG2, "LOG2")
        && constant(Q_LOG10, "LOG10")
        && constant(nanq(), "NAN")
        && constant(infq(), "INF");
    return ok ? 0 : -1;
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
    if (make_finfo() < 0)
        return NULL;

    numpy_module = PyImport_ImportModule("numpy");
    if (numpy_module == NULL)
        return NULL;

    PyArray_Descr *dtype = PyArray_DescrFromType(type_num);
    PyModule_AddObject(module, "dtype", (PyObject *)dtype);

    /* Casts need to be defined before ufuncs, because numpy >= 1.21 caches
     * casts/ufuncs in a way that is non-trivial... one should consider casts
     * to be "more basic".
     * See: https://github.com/numpy/numpy/issues/20009
     */
    if (register_casts() < 0)
        return NULL;
    if (register_ufuncs() < 0)
        return NULL;
    if (register_dtype_in_dicts() < 0)
        return NULL;
    if (register_constants() < 0)
        return NULL;

    /* Module is ready */
    return module;
}
