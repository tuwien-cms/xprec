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

#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_TYPE)
static inline void _Py_SET_TYPE(PyObject *ob, PyTypeObject *type)
{ ob->ob_type = type; }
#define Py_SET_TYPE(ob, type) _Py_SET_TYPE((PyObject*)(ob), type)
#endif

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
                *out = nanw();
            Py_XDECREF(arr);
        }
    } else {
        *out = nanw();
        PyErr_Format(PyExc_TypeError,
            "Cannot cast instance of %s to ddouble scalar",
            arg->ob_type->tp_name);
    }
    return !PyErr_Occurred();
}

static PyObject* PyDDouble_New(PyTypeObject *type, PyObject *args, PyObject *kwds)
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
    MARK_UNUSED(type);
    MARK_UNUSED(kwds);
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

PYWRAP_UNARY(PyDDouble_Positive, posw)
PYWRAP_UNARY(PyDDouble_Negative, negw)
PYWRAP_UNARY(PyDDouble_Absolute, absw)

PYWRAP_BINARY(PyDDouble_Add, addww, nb_add)
PYWRAP_BINARY(PyDDouble_Subtract, subww, nb_subtract)
PYWRAP_BINARY(PyDDouble_Multiply, mulww, nb_multiply)
PYWRAP_BINARY(PyDDouble_Divide, divww, nb_true_divide)

PYWRAP_INPLACE(PyDDouble_InPlaceAdd, addww)
PYWRAP_INPLACE(PyDDouble_InPlaceSubtract, subww)
PYWRAP_INPLACE(PyDDouble_InPlaceMultiply, mulww)
PYWRAP_INPLACE(PyDDouble_InPlaceDivide, divww)

static int PyDDouble_Nonzero(PyObject* _x)
{
    ddouble x = PyDDouble_Unwrap(_x);
    return !(x.hi == 0);
}

static PyObject* PyDDouble_RichCompare(PyObject* _x, PyObject* _y, int op)
{
    ddouble x, y;
    if (!PyDDouble_Cast(_x, &x) || !PyDDouble_Cast(_y, &y))
        return PyGenericArrType_Type.tp_richcompare(_x, _y, op);

    bool result;
    switch (op) {
    case Py_LT:
        result = lessww(x, y);
        break;
    case Py_LE:
        result = lessequalww(x, y);
        break;
    case Py_EQ:
        result = equalww(x, y);
        break;
    case Py_NE:
        result = notequalww(x, y);
        break;
    case Py_GT:
        result = greaterww(x, y);
        break;
    case Py_GE:
        result = greaterequalww(x, y);
        break;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Invalid op type");
        return NULL;
    }
    return PyBool_FromLong(result);
}

static Py_hash_t PyDDouble_Hash(PyObject *_x)
{
    ddouble x = PyDDouble_Unwrap(_x);

    int exp;
    double mantissa;
    mantissa = frexp(x.hi, &exp);
    return (Py_hash_t)(LONG_MAX * mantissa) + exp;
}

static PyObject *PyDDouble_Str(PyObject *self)
{
    char out[200];
    ddouble x = PyDDouble_Unwrap(self);
    snprintf(out, 200, "%.16g", x.hi);
    return PyUnicode_FromString(out);
}

static PyObject *PyDDouble_Repr(PyObject *self)
{
    char out[200];
    ddouble x = PyDDouble_Unwrap(self);
    snprintf(out, 200, "ddouble(%.16g+%.16g)", x.hi, x.lo);
    return PyUnicode_FromString(out);
}

static PyObject *PyDDoubleGetFinfo(PyObject *self, PyObject *_dummy)
{
    Py_INCREF(pyddouble_finfo);
    return pyddouble_finfo;
    MARK_UNUSED(self);
    MARK_UNUSED(_dummy);
}

static int make_ddouble_type()
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
        {NULL, NULL, 0, NULL}
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
        {"dtype",  T_OBJECT_EX, offsetof(PyDDoubleFInfo, dtype),  READONLY,
                   "underlying dtype object"},
        {"bits",   T_INT,       offsetof(PyDDoubleFInfo, bits),   READONLY,
                   "storage size of object in bits"},
        {"max",    T_OBJECT_EX, offsetof(PyDDoubleFInfo, max),    READONLY,
                   "largest positive number"},
        {"min",    T_OBJECT_EX, offsetof(PyDDoubleFInfo, min),    READONLY,
                   "largest negative number"},
        {"eps",    T_OBJECT_EX, offsetof(PyDDoubleFInfo, eps),    READONLY,
                   "machine epsilon"},
        {"nexp",   T_INT,       offsetof(PyDDoubleFInfo, nexp),   READONLY,
                   "number of bits in exponent"},
        {"nmant",  T_INT,       offsetof(PyDDoubleFInfo, nmant),  READONLY,
                   "number of bits in mantissa"},
        {"machar", T_OBJECT_EX, offsetof(PyDDoubleFInfo, machar), READONLY,
                   "machar object (unused)"},
        {NULL, 0, 0, 0, NULL}
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

    if (lessww(a, b))
        return -1;
    if (greaterww(a, b))
        return 1;
    if (isnanw(b))
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
    return !iszerow(x);
    MARK_UNUSED(arr);
}

static int NPyDDouble_Fill(void *_buffer, npy_intp ii, void *arr)
{
    // Fill with linear array
    ddouble *buffer = (ddouble *)_buffer;
    if (ii < 2)
        return -1;

    ddouble curr = buffer[1];
    ddouble step = subww(curr, buffer[0]);
    for (npy_intp i = 2; i != ii; ++i) {
        curr = addww(curr, step);
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
        out = addww(out, mulww(in1, in2));
    }
    *(ddouble *)_out = out;
    MARK_UNUSED(arr);
}

static int NPyDDouble_ArgMax(void *_data, npy_intp n, npy_intp *max_ind,
                             void *arr)
{
    ddouble *data = (ddouble *)_data;
    ddouble max_val = negw(infw());
    for (npy_intp i = 0; i < n; ++i) {
        if (greaterww(data[i], max_val)) {
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
    ddouble min_val = infw();
    for (npy_intp i = 0; i < n; ++i) {
        if (lessww(data[i], min_val)) {
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
    Py_SET_TYPE(&ddouble_dtype, &PyArrayDescr_Type);

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
        ddouble *to = (ddouble *)_to;                                \
        const from_type *from = (const from_type *)_from;            \
        for (npy_intp i = 0; i < n; ++i) {                           \
            double hi = from[i];                                     \
            double lo = from[i] - (from_type) hi;                    \
            to[i] = (ddouble){hi, lo};                               \
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

#define NPY_CAST_TO_I64(func, to_type)                               \
    static void func(void *_from, void *_to, npy_intp n,             \
                     void *_arr_from, void *_arr_to)                 \
    {                                                                \
        to_type *to = (to_type *)_to;                                \
        const ddouble *from = (const ddouble *)_from;                \
        for (npy_intp i = 0; i < n; ++i)                             \
            to[i] = (to_type) from[i].hi + (to_type) from[i].lo;     \
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
NPY_CAST_TO(to_uint8, uint8_t)
NPY_CAST_TO(to_uint16, uint16_t)
NPY_CAST_TO(to_uint32, uint32_t)

// These casts can be made more accurate
NPY_CAST_TO_I64(to_int64, int64_t)
NPY_CAST_TO_I64(to_uint64, uint64_t)


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

#define ULOOP_MODF(func_name, inner_func, type_out, type_a, type_b)     \
    static void func_name(char **args, const npy_intp *dimensions,      \
                          const npy_intp* steps, void *data)            \
    {                                                                   \
        const npy_intp n = dimensions[0];                               \
        const npy_intp as = steps[0] / sizeof(type_a),                  \
                       bs = steps[1] / sizeof(type_b),                  \
                       os = steps[2] / sizeof(type_out);                \
        const type_a *a = (const type_a *)args[0];                      \
        type_b *b = (type_b *)args[2];                                  \
        type_out *out = (type_out *)args[1];                            \
                                                                        \
        for (npy_intp i = 0; i < n; ++i) {                              \
            out[i * os] = inner_func(a[i * as], &b[i * bs]);            \
        }                                                               \
        MARK_UNUSED(data);                                              \
    }

ULOOP_BINARY(u_addwd, addwd, ddouble, ddouble, double)
ULOOP_BINARY(u_subwd, subwd, ddouble, ddouble, double)
ULOOP_BINARY(u_mulwd, mulwd, ddouble, ddouble, double)
ULOOP_BINARY(u_divwd, divwd, ddouble, ddouble, double)
ULOOP_BINARY(u_adddw, adddw, ddouble, double, ddouble)
ULOOP_BINARY(u_subdw, subdw, ddouble, double, ddouble)
ULOOP_BINARY(u_muldw, muldw, ddouble, double, ddouble)
ULOOP_BINARY(u_divdw, divdw, ddouble, double, ddouble)
ULOOP_BINARY(u_addww, addww, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_subww, subww, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_mulww, mulww, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_divww, divww, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_copysignww, copysignww, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_copysignwd, copysignwd, ddouble, ddouble, double)
ULOOP_BINARY(u_copysigndw, copysigndw, ddouble, double, ddouble)
ULOOP_BINARY(u_equalww, equalww, bool, ddouble, ddouble)
ULOOP_BINARY(u_notequalww, notequalww, bool, ddouble, ddouble)
ULOOP_BINARY(u_greaterww, greaterww, bool, ddouble, ddouble)
ULOOP_BINARY(u_lessww, lessww, bool, ddouble, ddouble)
ULOOP_BINARY(u_greaterequalww, greaterww, bool, ddouble, ddouble)
ULOOP_BINARY(u_lessequalww, lessww, bool, ddouble, ddouble)
ULOOP_BINARY(u_equalwd, equalwd, bool, ddouble, double)
ULOOP_BINARY(u_notequalwd, notequalwd, bool, ddouble, double)
ULOOP_BINARY(u_greaterwd, greaterwd, bool, ddouble, double)
ULOOP_BINARY(u_lesswd, lesswd, bool, ddouble, double)
ULOOP_BINARY(u_greaterequalwd, greaterequalwd, bool, ddouble, double)
ULOOP_BINARY(u_lessequalwd, lessequalwd, bool, ddouble, double)
ULOOP_BINARY(u_equaldw, equaldw, bool, double, ddouble)
ULOOP_BINARY(u_notequaldw, notequaldw, bool, double, ddouble)
ULOOP_BINARY(u_greaterdw, greaterdw, bool, double, ddouble)
ULOOP_BINARY(u_lessdw, lessdw, bool, double, ddouble)
ULOOP_BINARY(u_greaterequaldw, greaterequaldw, bool, double, ddouble)
ULOOP_BINARY(u_lessequaldw, lessequaldw, bool, double, ddouble)
ULOOP_BINARY(u_fminww, fminww, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_fmaxww, fmaxww, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_fminwd, fminwd, ddouble, ddouble, double)
ULOOP_BINARY(u_fmaxwd, fmaxwd, ddouble, ddouble, double)
ULOOP_BINARY(u_fmindw, fmindw, ddouble, double, ddouble)
ULOOP_BINARY(u_fmaxdw, fmaxdw, ddouble, double, ddouble)
ULOOP_BINARY(u_atan2wd, atan2wd, ddouble, ddouble, double)
ULOOP_BINARY(u_atan2dw, atan2dw, ddouble, double, ddouble)
ULOOP_BINARY(u_atan2ww, atan2ww, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_powwd, powwd, ddouble, ddouble, double)
ULOOP_BINARY(u_powdw, powdw, ddouble, double, ddouble)
ULOOP_BINARY(u_powww, powww, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_hypotww, hypotww, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_hypotdw, hypotdw, ddouble, double, ddouble)
ULOOP_BINARY(u_hypotwd, hypotwd, ddouble, ddouble, double)
ULOOP_BINARY(u_ldexpwi, ldexpwi, ddouble, ddouble, int)
ULOOP_MODF(u_modfww, modfww, ddouble, ddouble, ddouble)
ULOOP_UNARY(u_signbitw, signbitw, bool, ddouble)
ULOOP_UNARY(u_signw, signw, ddouble, ddouble)
ULOOP_UNARY(u_isfinitew, isfinitew, bool, ddouble)
ULOOP_UNARY(u_isinfw, isinfw, bool, ddouble)
ULOOP_UNARY(u_isnanw, isnanw, bool, ddouble)
ULOOP_UNARY(u_negw, negw, ddouble, ddouble)
ULOOP_UNARY(u_posw, posw, ddouble, ddouble)
ULOOP_UNARY(u_absw, absw, ddouble, ddouble)
ULOOP_UNARY(u_reciprocalw, reciprocalw, ddouble, ddouble)
ULOOP_UNARY(u_sqrw, sqrw, ddouble, ddouble)
ULOOP_UNARY(u_roundw, roundw, ddouble, ddouble)
ULOOP_UNARY(u_floorw, floorw, ddouble, ddouble)
ULOOP_UNARY(u_ceilw, ceilw, ddouble, ddouble)
ULOOP_UNARY(u_sqrtw, sqrtw, ddouble, ddouble)
ULOOP_UNARY(u_expw, expw, ddouble, ddouble)
ULOOP_UNARY(u_expm1w, expm1w, ddouble, ddouble)
ULOOP_UNARY(u_logw, logw, ddouble, ddouble)
ULOOP_UNARY(u_sinw, sinw, ddouble, ddouble)
ULOOP_UNARY(u_cosw, cosw, ddouble, ddouble)
ULOOP_UNARY(u_tanw, tanw, ddouble, ddouble)
ULOOP_UNARY(u_atanw, atanw, ddouble, ddouble)
ULOOP_UNARY(u_acosw, acosw, ddouble, ddouble)
ULOOP_UNARY(u_asinw, asinw, ddouble, ddouble)
ULOOP_UNARY(u_atanhw, atanhw, ddouble, ddouble)
ULOOP_UNARY(u_acoshw, acoshw, ddouble, ddouble)
ULOOP_UNARY(u_asinhw, asinhw, ddouble, ddouble)
ULOOP_UNARY(u_sinhw, sinhw, ddouble, ddouble)
ULOOP_UNARY(u_coshw, coshw, ddouble, ddouble)
ULOOP_UNARY(u_tanhw, tanhw, ddouble, ddouble)

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

static int register_ldexp(PyUFuncGenericFunction func, int ret_dtype,
                          const char *name)
{
    PyUFuncObject *ufunc;
    int *arg_types = NULL, retcode = 0;

    ufunc = (PyUFuncObject *)PyObject_GetAttrString(numpy_module, name);
    if (ufunc == NULL) goto error;

    arg_types = PyMem_New(int, 3);
    if (arg_types == NULL) goto error;

    arg_types[0] = type_num;
    arg_types[1] = NPY_INTP;
    arg_types[2] = ret_dtype;
    retcode = PyUFunc_RegisterLoopForType(ufunc, type_num,
                                          func, arg_types, NULL);
    if (retcode < 0) goto error;
    return true;

error:
    return false;
}

static int register_modf(PyUFuncGenericFunction func, int ret_dtype,
                          const char *name)
{
    PyUFuncObject *ufunc;
    int *arg_types = NULL, retcode = 0;

    ufunc = (PyUFuncObject *)PyObject_GetAttrString(numpy_module, name);
    if (ufunc == NULL) goto error;

    arg_types = PyMem_New(int, 4);
    if (arg_types == NULL) goto error;

    arg_types[0] = type_num;
    arg_types[1] = type_num;
    arg_types[2] = ret_dtype;
    arg_types[3] = ret_dtype;
    retcode = PyUFunc_RegisterLoopForType(ufunc, type_num,
                                          func, arg_types, NULL);
    if (retcode < 0) goto error;
    return true;

error:
    return false;
}

static int register_ufuncs()
{
    bool ok = register_unary(u_negw, type_num, "negative")
        && register_unary(u_posw, type_num, "positive")
        && register_unary(u_absw, type_num, "absolute")
        && register_unary(u_reciprocalw, type_num, "reciprocal")
        && register_unary(u_sqrw, type_num, "square")
        && register_unary(u_sqrtw, type_num, "sqrt")
        && register_unary(u_signbitw, NPY_BOOL, "signbit")
        && register_unary(u_isfinitew, NPY_BOOL, "isfinite")
        && register_unary(u_isinfw, NPY_BOOL, "isinf")
        && register_unary(u_isnanw, NPY_BOOL, "isnan")
        && register_unary(u_roundw, type_num, "rint")
        && register_unary(u_floorw, type_num, "floor")
        && register_unary(u_ceilw, type_num, "ceil")
        && register_unary(u_expw, type_num, "exp")
        && register_unary(u_expm1w, type_num, "expm1")
        && register_unary(u_logw, type_num, "log")
        && register_unary(u_sinw, type_num, "sin")
        && register_unary(u_cosw, type_num, "cos")
        && register_unary(u_tanw, type_num, "tan")
        && register_unary(u_atanw, type_num, "arctan")
        && register_unary(u_acosw, type_num, "arccos")
        && register_unary(u_asinw, type_num, "arcsin")
        && register_unary(u_atanhw, type_num, "arctanh")
        && register_unary(u_acoshw, type_num, "arccosh")
        && register_unary(u_asinhw, type_num, "arcsinh")
        && register_unary(u_sinhw, type_num, "sinh")
        && register_unary(u_coshw, type_num, "cosh")
        && register_unary(u_tanhw, type_num, "tanh")
        && register_unary(u_signw, type_num, "sign")
        && register_ldexp(u_ldexpwi, type_num, "ldexp")
        && register_modf(u_modfww, type_num, "modf")
        && register_binary(u_adddw, u_addwd, u_addww, type_num, "add")
        && register_binary(u_subdw, u_subwd, u_subww, type_num, "subtract")
        && register_binary(u_muldw, u_mulwd, u_mulww, type_num, "multiply")
        && register_binary(u_divdw, u_divwd, u_divww, type_num, "true_divide")
        && register_binary(u_powdw, u_powwd, u_powww, type_num, "power")
        && register_binary(u_equaldw, u_equalwd, u_equalww, NPY_BOOL, "equal")
        && register_binary(u_notequaldw, u_notequalwd, u_notequalww, NPY_BOOL,
                           "not_equal")
        && register_binary(u_greaterdw, u_greaterwd, u_greaterww, NPY_BOOL, "greater")
        && register_binary(u_lessdw, u_lesswd, u_lessww, NPY_BOOL, "less")
        && register_binary(u_greaterequaldw, u_greaterequalwd, u_greaterequalww,
                           NPY_BOOL, "greater_equal")
        && register_binary(u_lessequaldw, u_lessequalwd, u_lessequalww, NPY_BOOL,
                           "less_equal")
        && register_binary(u_fmindw, u_fminwd, u_fminww, type_num, "fmin")
        && register_binary(u_fmaxdw, u_fmaxwd, u_fmaxww, type_num, "fmax")
        && register_binary(u_fmindw, u_fminwd, u_fminww, type_num, "minimum")
        && register_binary(u_fmaxdw, u_fmaxwd, u_fmaxww, type_num, "maximum")
        && register_binary(u_atan2dw, u_atan2wd, u_atan2ww, type_num, "arctan2")
        && register_binary(u_copysigndw, u_copysignwd, u_copysignww, type_num,
                           "copysign")
        && register_binary(u_hypotdw, u_hypotwd, u_hypotww, type_num, "hypot");
    return ok ? 0 : -1;
}

static int register_dtype_in_dicts()
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

static PyObject *make_module()
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
        && constant(nanw(), "NAN")
        && constant(infw(), "INF");
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
