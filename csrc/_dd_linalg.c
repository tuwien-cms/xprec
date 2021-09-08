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
    const npy_intp san = steps[0], sbn = steps[1], scn = steps[2],
                   sai = steps[3], saj = steps[4], sbj = steps[5],
                   sbk = steps[6], sci = steps[7], sck = steps[8];
    char *_a = args[0], *_b = args[1], *_c = args[2];

    for (npy_intp n = 0; n != nn; ++n, _a += san, _b += sbn, _c += scn) {
        for (npy_intp i = 0; i != ii; ++i) {
            for (npy_intp k = 0; k != kk; ++k) {
                ddouble val = Q_ZERO;
                for (npy_intp j = 0; j != jj; ++j) {
                    const ddouble *a_ij =
                            (const ddouble *) (_a + i * sai + j * saj);
                    const ddouble *b_jk =
                            (const ddouble *) (_b + j * sbj + k * sbk);
                    val = addqq(val, mulqq(*a_ij, *b_jk));

                }
                ddouble *c_ik = (ddouble *) (_c + i * sci + k * sck);
                *c_ik = val;
            }
        }
    }
    MARK_UNUSED(data);
}

/*************************** More complicated ***********************/

static void u_normq(char **args, const npy_intp *dims,
                    const npy_intp* steps, void *data)
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

static void u_householderq(char **args, const npy_intp *dims,
                           const npy_intp* steps, void *data)
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

static void u_givensq(char **args, const npy_intp *dims, const npy_intp* steps,
                      void *data)
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

static void u_svd_tri2x2(char **args, const npy_intp *dims,
                         const npy_intp* steps, void *data)
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
        ddouble f = *(ddouble *) _a;
        ddouble z = *(ddouble *) (_a + sai);
        ddouble g = *(ddouble *) (_a + saj);
        ddouble h = *(ddouble *) (_a + sai + saj);

        ddouble smin, smax, cu, su, cv, sv;
        if (!iszeroq(z)) {
            fprintf(stderr, "svd_tri2x2: matrix is not upper triagonal\n");
            smin = smax = cu = su = cv = sv = nanq();
        } else {
            svd_tri2x2(f, g, h, &smin, &smax, &cv, &sv, &cu, &su);
        }

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

static void u_svvals_tri2x2(char **args, const npy_intp *dims,
                            const npy_intp* steps, void *data)
{
    // signature (n;2,2)->(n;2)
    const npy_intp nn = dims[0];
    const npy_intp san = steps[0], sbn = steps[1], sai = steps[2],
                   saj = steps[3], sbi = steps[4];
    char *_a = args[0], *_b = args[1];

    for (npy_intp n = 0; n != nn; ++n, _a += san, _b += sbn) {
        ddouble f = *(ddouble *) _a;
        ddouble z = *(ddouble *) (_a + sai);
        ddouble g = *(ddouble *) (_a + saj);
        ddouble h = *(ddouble *) (_a + sai + saj);

        ddouble smin, smax;
        if (!iszeroq(z)) {
            fprintf(stderr, "svd_tri2x2: matrix is not upper triagonal\n");
            smin = smax = nanq();
        } else {
            svd_tri2x2(f, g, h, &smin, &smax, NULL, NULL, NULL, NULL);
        }

        *(ddouble *)_b = smax;
        *(ddouble *)(_b + sbi) = smin;
    }
    MARK_UNUSED(data);
}

/* ----------------------- Python stuff -------------------------- */

static const char DDOUBLE_WRAP = NPY_CDOUBLE;

static void gufunc(PyObject *module_dict, PyUFuncGenericFunction uloop,
                   int nin, int nout, const char *signature, const char *name,
                   const char *docstring)
{
    PyObject *ufunc;
    PyUFuncGenericFunction* loops = PyMem_New(PyUFuncGenericFunction, 1);
    char *dtypes = PyMem_New(char, nin + nout);
    void **data = PyMem_New(void *, 1);

    loops[0] = uloop;
    data[0] = NULL;
    for (int i = 0; i != nin + nout; ++i)
        dtypes[i] = DDOUBLE_WRAP;

    ufunc = PyUFunc_FromFuncAndDataAndSignature(
                loops, data, dtypes, 1, nin, nout, PyUFunc_None, name,
                docstring, 0, signature);
    PyDict_SetItemString(module_dict, name, ufunc);
    Py_DECREF(ufunc);
}

PyMODINIT_FUNC PyInit__dd_linalg(void)
{
    // Defitions
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

    gufunc(module_dict, u_normq, 1, 1, "(i)->()",
           "norm", "Vector 2-norm");
    gufunc(module_dict, u_matmulq, 2, 1, "(i?,j),(j,k?)->(i?,k?)",
           "matmul", "Matrix multiplication");
    gufunc(module_dict, u_givensq, 1, 2, "(2)->(2),(2,2)",
           "givens", "Generate Givens rotation");
    gufunc(module_dict, u_householderq, 1, 2, "(i)->(),(i)",
           "householder", "Generate Householder reflectors");
    gufunc(module_dict, u_svd_tri2x2, 1, 3, "(2,2)->(2,2),(2),(2,2)",
           "svd_tri2x2", "SVD of upper triangular 2x2 problem");
    gufunc(module_dict, u_svvals_tri2x2, 1, 1, "(2,2)->(2)",
           "svvals_tri2x2", "singular values of upper triangular 2x2 problem");

    /* Make dtype */
    dtype = PyArray_DescrFromType(DDOUBLE_WRAP);
    PyDict_SetItemString(module_dict, "dtype", (PyObject *)dtype);

    /* Module is ready */
    return module;
}
