/* Python extension module for linear algebra functions.
 *
 * Copyright (C) 2021 Markus Wallerberger and others
 * SPDX-License-Identifier: MIT
 */
#include "Python.h"
#include "math.h"
#include "stdio.h"

#include "dd_arith.h"
#include "dd_linalg.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

/**
 * Allows parameter to be marked unused
 */
#define MARK_UNUSED(x)  do { (void)(x); } while(false)

/************************ Linear algebra ***************************/

static void u_matmulq(char **args, const npy_intp *dims, const npy_intp* steps,
                      void *data)
{
    // signature (n;i,j),(n;j,k)->(n;i,k)
    const npy_intp nn = dims[0], ii = dims[1], jj = dims[2], kk = dims[3];
    const npy_intp _san = steps[0], _sbn = steps[1], _scn = steps[2],
                   _sai = steps[3], _saj = steps[4], _sbj = steps[5],
                   _sbk = steps[6], _sci = steps[7], _sck = steps[8];
    char *_a = args[0], *_b = args[1], *_c = args[2];

    const npy_intp sai = _sai / sizeof(ddouble), saj = _saj / sizeof(ddouble),
                   sbj = _sbj / sizeof(ddouble), sbk = _sbk / sizeof(ddouble),
                   sci = _sci / sizeof(ddouble), sck = _sck / sizeof(ddouble);

    for (npy_intp n = 0; n != nn; ++n, _a += _san, _b += _sbn, _c += _scn) {
        const ddouble *a = (const ddouble *)_a, *b = (const ddouble *)_b;
        ddouble *c = (ddouble *)_c;

        #pragma omp parallel for collapse(2)
        for (npy_intp i = 0; i < ii; ++i) {
            for (npy_intp k = 0; k < kk; ++k) {
                ddouble val = Q_ZERO, tmp;
                for (npy_intp j = 0; j < jj; ++j) {
                    tmp = mulqq(a[i * sai + j * saj], b[j * sbj + k * sbk]);
                    val = addqq(val, tmp);
                }
                c[i * sci + k * sck] = val;
            }
        }
    }
    MARK_UNUSED(data);
}

/****************************** Helper functions *************************/

static void ensure_inplace_2(
        char *in, char *out, npy_intp n1, npy_intp si1, npy_intp so1,
        npy_intp n2, npy_intp si2, npy_intp so2)
{
    if (in == out)
        return;

    char *in1 = in, *out1 = out;
    for (npy_intp i1 = 0; i1 != n1; ++i1, in1 += si1, out1 += so1) {
        char *in2 = in1, *out2 = out1;
        for (npy_intp i2 = 0; i2 != n2; ++i2, in2 += si2, out2 += so2) {
            char *inx = in2, *outx = out2;
            *(ddouble *)outx = *(ddouble *)inx;
        }
    }
}

static void ensure_inplace_3(
        char *in, char *out, npy_intp n1, npy_intp si1, npy_intp so1,
        npy_intp n2, npy_intp si2, npy_intp so2, npy_intp n3, npy_intp si3,
        npy_intp so3)
{
    if (in == out)
        return;

    char *in1 = in, *out1 = out;
    for (npy_intp i1 = 0; i1 != n1; ++i1, in1 += si1, out1 += so1) {
        char *in2 = in1, *out2 = out1;
        for (npy_intp i2 = 0; i2 != n2; ++i2, in2 += si2, out2 += so2) {
            char *in3 = in2, *out3 = out2;
            for (npy_intp i3 = 0; i3 != n3; ++i3, in3 += si3, out3 += so3) {
                char *inx = in3, *outx = out3;
                *(ddouble *)outx = *(ddouble *)inx;
            }
        }
    }
}

/*************************** More complicated ***********************/

static void u_normq(
    char **args, const npy_intp *dims, const npy_intp* steps, void *data)
{
   // signature (n;i)->(n;)
    const npy_intp nn = dims[0], ii = dims[1];
    const npy_intp san = steps[0], sbn = steps[1], _sai = steps[2];
    char *_a = args[0], *_b = args[1];

    for (npy_intp n = 0; n != nn; ++n, _a += san, _b += sbn) {
        *(ddouble *)_b = normq((const ddouble *)_a, ii, _sai / sizeof(ddouble));
    }
    MARK_UNUSED(data);
}

static void u_householderq(
    char **args, const npy_intp *dims, const npy_intp* steps, void *data)
{
    // signature (n;i)->(n;),(n;i)
    const npy_intp nn = dims[0], ii = dims[1];
    const npy_intp _san = steps[0], _sbn = steps[1], _scn = steps[2],
                   _sai = steps[3], _sci = steps[4];
    char *_a = args[0], *_b = args[1], *_c = args[2];

    for (npy_intp n = 0; n != nn; ++n, _a += _san, _b += _sbn, _c += _scn) {
        *(ddouble *)_b = householderq(
                (const ddouble *)_a, (ddouble *)_c, ii,
                _sai / sizeof(ddouble), _sci / sizeof(ddouble));
    }
    MARK_UNUSED(data);
}

static void u_rank1updateq(
    char **args, const npy_intp *dims, const npy_intp* steps, void *data)
{
    // signature (n;i,j),(n;i),(n;j)->(n;i,j)
    const npy_intp nn = dims[0], ii = dims[1], jj = dims[2];
    const npy_intp _san = steps[0], _sbn = steps[1], _scn = steps[2],
                   _sdn = steps[3], _sai = steps[4], _saj = steps[5],
                   _sbi = steps[6], _scj = steps[7], _sdi = steps[8],
                   _sdj = steps[9];
    char *_a = args[0], *_b = args[1], *_c = args[2], *_d = args[3];

    ensure_inplace_3(_a, _d, nn, _san, _sdn, ii, _sai, _sdi, jj, _saj, _sdj);
    for (npy_intp n = 0; n != nn; ++n, _a += _san, _b += _sbn, _c += _scn) {
        rank1updateq(
            (ddouble *)_d, _sai / sizeof(ddouble), _saj / sizeof(ddouble),
            (const ddouble *)_b, _sbi / sizeof(ddouble),
            (const ddouble *)_c, _scj / sizeof(ddouble), ii, jj);
    }
    MARK_UNUSED(data);
}

static void u_jacobisweepq(
    char **args, const npy_intp *dims, const npy_intp* steps, void *data)
{
    // signature (n;i,j),(n;i=j,j)->(n;i,j),(n;i=j,j);(n,)
    const npy_intp nn = dims[0], ii = dims[1], jj = dims[2];
    const npy_intp _san = steps[0], _sbn = steps[1],  _scn = steps[2],
                   _sdn = steps[3], _sen = steps[4],  _sai = steps[5],
                   _saj = steps[6], _sbi = steps[7],  _sbj = steps[8],
                   _sci = steps[9], _scj = steps[10], _sdi = steps[11],
                   _sdj = steps[12];
    char *_a = args[0], *_b = args[1], *_c = args[2], *_d = args[3],
         *_e = args[4];

    ensure_inplace_3(_a, _c, nn, _san, _scn, ii, _sai, _sci, jj, _saj, _scj);
    ensure_inplace_3(_b, _d, nn, _sbn, _sdn, jj, _sbi, _sdi, jj, _sbj, _sdj);
    for (npy_intp n = 0; n != nn; ++n, _c += _scn, _d += _sdn, _e += _sen) {
        ddouble *c = (ddouble *)_c, *d = (ddouble *)_d, *e = (ddouble *)_e;
        const npy_intp
                sci = _sci / sizeof(ddouble), scj = _scj / sizeof(ddouble),
                sdi = _sdi / sizeof(ddouble), sdj = _sdj / sizeof(ddouble);

        *e = jacobi_sweep(c, sci, scj, d, sdi, sdj, ii, jj);
    }
    MARK_UNUSED(data);
}

static void u_givensq(
    char **args, const npy_intp *dims, const npy_intp* steps, void *data)
{
    // signature (n;2)->(n;2),(n;2,2)
    const npy_intp nn = dims[0];
    const npy_intp san = steps[0], sbn = steps[1], scn = steps[2],
                   sai = steps[3], sbi = steps[4], sci = steps[5],
                   scj = steps[6];
    char *_a = args[0], *_b = args[1], *_c = args[2];

    for (npy_intp n = 0; n != nn; ++n, _a += san, _b += sbn, _c += scn) {
        ddouble f = *(ddouble *) _a;
        ddouble g = *(ddouble *) (_a + sai);

        ddouble c, s, r;
        givensq(f, g, &c, &s, &r);

        *(ddouble *)_b = r;
        *(ddouble *)(_b + sbi) = Q_ZERO;
        *(ddouble *)_c = c;
        *(ddouble *)(_c + scj) = s;
        *(ddouble *)(_c + sci) = negq(s);
        *(ddouble *)(_c + sci + scj) = c;
    }
    MARK_UNUSED(data);
}

static void u_givens_seqq(
    char **args, const npy_intp *dims, const npy_intp* steps, void *data)
{
    // signature (n;i,2),(n;i,j)->(n;i,j)
    const npy_intp nn = dims[0], ii = dims[1], jj = dims[3];
    const npy_intp _san = steps[0], _sbn = steps[1], _scn = steps[2],
                   _sai = steps[3], _saq = steps[4], _sbi = steps[5],
                   _sbj = steps[6], _sci = steps[7], _scj = steps[8];
    char *_a = args[0], *_b = args[1], *_c = args[2];

    ensure_inplace_3(_b, _c, nn, _sbn, _scn, ii, _sbi, _sci, jj, _sbj, _scj);
    for (npy_intp n = 0; n != nn; ++n, _a += _san, _c += _scn) {
        /* The rotation are interdependent, so we splice the array in
         * the other direction.
         */
        #pragma omp parallel for
        for (npy_intp j = 0; j < jj; ++j) {
            for (npy_intp i = 0; i < ii - 1; ++i) {
                ddouble *c_x = (ddouble *)(_c + i *_sci + j * _scj);
                ddouble *c_y = (ddouble *)(_c + (i + 1) *_sci + j * _scj);
                ddouble g_cos = *(ddouble *)(_a + i * _sai);
                ddouble g_sin = *(ddouble *)(_a + i * _sai + _saq);
                lmul_givensq(c_x, c_y, g_cos, g_sin, *c_x, *c_y);
            }
        }
    }
    MARK_UNUSED(data);
}

static void u_golub_kahan_chaseq(
    char **args, const npy_intp *dims, const npy_intp* steps, void *data)
{
    // signature (n;i),(n;i)->(n;i),(n;i),(n;i,4)
    const npy_intp nn = dims[0], ii = dims[1];
    const npy_intp _san = steps[0], _sbn = steps[1],  _scn = steps[2],
                   _sdn = steps[3], _sen = steps[4],  _sai = steps[5],
                   _sbi = steps[6], _sci = steps[7],  _sdi = steps[8],
                   _sei = steps[9], _se4 = steps[10];
    char *_a = args[0], *_b = args[1], *_c = args[2], *_d = args[3],
         *_e = args[4];

    ensure_inplace_2(_a, _c, nn, _san, _scn, ii, _sai, _sci);
    ensure_inplace_2(_b, _d, nn, _sbn, _sdn, ii, _sbi, _sdi);
    if (_se4 != sizeof(ddouble) || _sei != 4 * sizeof(ddouble)) {
        fprintf(stderr, "rot is not contiguous, but needs to be");
        return;
    }

    for (npy_intp n = 0; n != nn; ++n, _c += _scn, _d += _sdn, _e += _sen) {
        golub_kahan_chaseq((ddouble *)_c, _sci / sizeof(ddouble),
                           (ddouble *)_d, _sdi / sizeof(ddouble),
                           ii, (ddouble *)_e);
    }
    MARK_UNUSED(data);
}

static void u_svd_2x2(
    char **args, const npy_intp *dims, const npy_intp* steps, void *data)
{
    // signature (n;2,2)->(n;2,2),(n;2),(n;2,2)
    const npy_intp nn = dims[0];
    const npy_intp san = steps[0], sbn = steps[1], scn = steps[2],
                   sdn = steps[3], sai = steps[4], saj = steps[5],
                   sbi = steps[6], sbj = steps[7], sci = steps[8],
                   sdi = steps[9], sdj = steps[10];
    char *_a = args[0], *_b = args[1], *_c = args[2], *_d = args[3];

    for (npy_intp n = 0; n != nn;
                ++n, _a += san, _b += sbn, _c += scn, _d += sdn) {
        ddouble a11 = *(ddouble *) _a;
        ddouble a12 = *(ddouble *) (_a + saj);
        ddouble a21 = *(ddouble *) (_a + sai);
        ddouble a22 = *(ddouble *) (_a + sai + saj);

        ddouble smin, smax, cu, su, cv, sv;
        svd_2x2(a11, a12, a21, a22, &smin, &smax, &cv, &sv, &cu, &su);

        *(ddouble *)_b = cu;
        *(ddouble *)(_b + sbj) = negq(su);
        *(ddouble *)(_b + sbi) = su;
        *(ddouble *)(_b + sbi + sbj) = cu;

        *(ddouble *)_c = smax;
        *(ddouble *)(_c + sci) = smin;

        *(ddouble *)_d = cv;
        *(ddouble *)(_d + sdj) = sv;
        *(ddouble *)(_d + sdi) = negq(sv);
        *(ddouble *)(_d + sdi + sdj) = cv;
    }
    MARK_UNUSED(data);
}

static void u_svvals_2x2(
    char **args, const npy_intp *dims, const npy_intp* steps, void *data)
{
    // signature (n;2,2)->(n;2)
    const npy_intp nn = dims[0];
    const npy_intp san = steps[0], sbn = steps[1], sai = steps[2],
                   saj = steps[3], sbi = steps[4];
    char *_a = args[0], *_b = args[1];

    for (npy_intp n = 0; n != nn; ++n, _a += san, _b += sbn) {
        ddouble a11 = *(ddouble *) _a;
        ddouble a12 = *(ddouble *) (_a + saj);
        ddouble a21 = *(ddouble *) (_a + sai);
        ddouble a22 = *(ddouble *) (_a + sai + saj);

        ddouble smin, smax;
        svd_2x2(a11, a12, a21, a22, &smin, &smax, NULL, NULL, NULL, NULL);

        *(ddouble *)_b = smax;
        *(ddouble *)(_b + sbi) = smin;
    }
    MARK_UNUSED(data);
}

/* ----------------------- Python stuff -------------------------- */

static PyObject *module;
static PyObject *numpy_module = NULL;
static int type_num;

static PyObject *make_module()
{
    static PyMethodDef no_methods[] = {
        {NULL, NULL, 0, NULL}    // No methods defined
    };
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_dd_linalg",
        NULL,
        -1,
        no_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };
    module = PyModule_Create(&module_def);
    return module;
}

static int import_ddouble_dtype()
{
    PyObject *dd_module = PyImport_ImportModule("xprec._dd_ufunc");
    if (dd_module == NULL)
        return -1;

    // Now, ddouble should be defined
    type_num = PyArray_TypeNumFromName("ddouble");
    if (type_num == NPY_NOTYPE)
        return -1;

    return 0;
}

static int gufunc(
        PyUFuncGenericFunction uloop, int nin, int nout,
        const char *signature, const char *name, const char *docstring,
        bool in_numpy)
{
    PyUFuncObject *ufunc = NULL;
    int *arg_types = NULL, retcode = 0;

    if (in_numpy) {
        ufunc = (PyUFuncObject *)PyObject_GetAttrString(numpy_module, name);
    } else {
        ufunc = (PyUFuncObject *)PyUFunc_FromFuncAndDataAndSignature(
                        NULL, NULL, NULL, 0, nin, nout, PyUFunc_None, name,
                        docstring, 0, signature);
    }
    if (ufunc == NULL) goto error;

    int *dtypes = PyMem_New(int, nin + nout);
    if (dtypes == NULL) goto error;

    for (int i = 0; i != nin + nout; ++i)
        dtypes[i] = type_num;

    retcode = PyUFunc_RegisterLoopForType(ufunc, type_num,
                                          uloop, arg_types, NULL);
    if (retcode < 0) goto error;

    return PyModule_AddObject(module, name, (PyObject *)ufunc);

error:
    if (!in_numpy)
        Py_XDECREF(ufunc);
    PyMem_Free(arg_types);
    return -1;
}

PyMODINIT_FUNC PyInit__dd_linalg(void)
{
    if (!make_module())
        return NULL;

    /* Initialize numpy things */
    import_array();
    import_umath();

    numpy_module = PyImport_ImportModule("numpy");
    if (numpy_module == NULL)
       return NULL;

    if (import_ddouble_dtype() < 0)
        return NULL;

    gufunc(u_normq, 1, 1, "(i)->()",
           "norm", "Vector 2-norm", false);
    gufunc(u_matmulq, 2, 1, "(i?,j),(j,k?)->(i?,k?)",
           "matmul", "Matrix multiplication", true);
    gufunc(u_givensq, 1, 2, "(2)->(2),(2,2)",
           "givens", "Generate Givens rotation", false);
    gufunc(u_givens_seqq, 2, 1, "(i,2),(i,j?)->(i,j?)",
           "givens_seq", "apply sequence of givens rotation to matrix", false);
    gufunc(u_householderq, 1, 2, "(i)->(),(i)",
           "householder", "Generate Householder reflectors", false);
    gufunc(u_rank1updateq, 3, 1, "(i,j),(i),(j)->(i,j)",
           "rank1update", "Perform rank-1 update of matrix", false);
    gufunc(u_svd_2x2, 1, 3, "(2,2)->(2,2),(2),(2,2)",
           "svd2x2", "SVD of upper triangular 2x2 problem", false);
    gufunc(u_svvals_2x2, 1, 1, "(2,2)->(2)",
           "svvals2x2", "singular values of upper triangular 2x2 problem", false);
    gufunc(u_jacobisweepq, 2, 3, "(i,j),(j,j)->(i,j),(j,j),()",
           "jacobi_sweep", "Perform sweep of one-sided Jacobi rotations", false);
    gufunc(u_golub_kahan_chaseq, 2, 3, "(i),(i)->(i),(i),(i,4)",
           "golub_kahan_chase", "bidiagonal chase procedure", false);

    /* Make dtype */
    PyArray_Descr *dtype = PyArray_DescrFromType(NPY_CDOUBLE);
    PyModule_AddObject(module, "dtype", (PyObject *)dtype);

    /* Module is ready */
    return module;
}
