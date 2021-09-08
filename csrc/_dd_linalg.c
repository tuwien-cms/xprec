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


// 2**500 and 2**(-500);
static const double LARGE = 3.273390607896142e+150;
static const double INV_LARGE = 3.054936363499605e-151;


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

static ddouble normq_scaled(const ddouble *x, long nn, long sxn,
                            double scaling)
{
    ddouble sum = Q_ZERO;
    for (long n = 0; n < nn; ++n, x += sxn) {
        ddouble curr = mul_pwr2(*x, scaling);
        sum = addqq(sum, sqrq(curr));
    };
    return mul_pwr2(sqrtq(sum), 1.0/scaling);
}

static ddouble normq(const ddouble *x, long nn, long sxn)
{
    ddouble sum = normq_scaled(x, nn, sxn, 1.0);

    // fall back to other routines in case of over/underflow
    if (sum.hi > LARGE)
        return normq_scaled(x, nn, sxn, INV_LARGE);
    else if (sum.hi < INV_LARGE)
        return normq_scaled(x, nn, sxn, LARGE);
    else
        return sum;
}

static void u_normq(char **args, const npy_intp *dims,
                    const npy_intp* steps, void *data)
{
   // signature (n;i)->(n;)
    const npy_intp nn = dims[0], ii = dims[1];
    const npy_intp san = steps[0], sbn = steps[1], _sai = steps[2];
    char *_a = args[0], *_b = args[1];

    const npy_intp sai = _sai / sizeof(ddouble);
    for (npy_intp n = 0; n != nn; ++n, _a += san, _b += sbn) {
        const ddouble *a = (const ddouble *)_a;
        ddouble *b = (ddouble *)_b;

        *b = normq(a, ii, sai);
    }
    MARK_UNUSED(data);
}

// static void u_householderq(char **args, const npy_intp *dims,
//                            const npy_intp* steps, void *data)
// {
//     // signature (n;i)->(n;1),(n;i)
//     const npy_intp nn = dims[0], ii = dims[1];
//     const npy_intp san = steps[0], sbn = steps[1], scn = steps[2],
//                    sai = steps[3], sci = steps[4];
//     char *_a = args[0], *_b = args[1], *_c = args[2];

//     for (npy_intp n = 0; n != nn; ++n, _a += san, _b += sbn, _c += scn) {
//         // compute norm of vector
//         const ddouble *a = (const ddouble *)_a;
//         ddouble norm = normq((const ddouble *)_a, ii, sai / sizeof(double));
//     }
//     MARK_UNUSED(data);
// }

static void givensq(ddouble f, ddouble g, ddouble *c, ddouble *s, ddouble *r)
{
    /* ACM Trans. Math. Softw. 28(2), 206, Alg 1 */
    if (iszeroq(g)) {
        *c = Q_ONE;
        *s = Q_ZERO;
        *r = f;
    } else if (iszeroq(f)) {
        *c = Q_ZERO;
        *s = (ddouble) {signbitq(g), 0.0};
        *r = absq(g);
    } else {
        *r = copysignqq(hypotqq(f, g), f);

        /* This may come at a slight loss of precision, however, we should
         * not really have to care ...
         */
        ddouble inv_r = reciprocalq(*r);
        *c = mulqq(f, inv_r);
        *s = mulqq(g, inv_r);
    }
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

static void svd_tri2x2(ddouble f, ddouble g, ddouble h, ddouble *smin,
                       ddouble *smax, ddouble *cv, ddouble *sv, ddouble *cu,
                       ddouble *su)
{
    ddouble fa = absq(f);
    ddouble ga = absq(g);
    ddouble ha = absq(h);
    bool compute_uv = cv != NULL;

    if (lessqq(fa, ha)) {
        // switch h <-> f, cu <-> sv, cv <-> su
        svd_tri2x2(h, g, f, smin, smax, su, cu, sv, cv);
        return;
    }
    if (iszeroq(ga)) {
        // already diagonal
        *smin = ha;
        *smax = fa;
        if (compute_uv) {
            *cu = Q_ONE;
            *su = Q_ZERO;
            *cv = Q_ONE;
            *sv = Q_ZERO;
        }
        return;
    }
    if (fa.hi < Q_EPS.hi * ga.hi) {
        // ga is very large
        *smax = ga;
        if (ha.hi > 1.0)
            *smin = divqq(fa, divqq(ga, ha));
        else
            *smin = mulqq(divqq(fa, ga), ha);
        if (compute_uv) {
            *cu = Q_ONE;
            *su = divqq(h, g);
            *cv = Q_ONE;
            *sv = divqq(f, g);
        }
        return;
    }
    // normal case
    ddouble fmh = subqq(fa, ha);
    ddouble d = divqq(fmh, fa);
    ddouble q = divqq(g, f);
    ddouble s = subdq(2.0, d);
    ddouble spq = hypotqq(q, s);
    ddouble dpq = hypotqq(d, q);
    ddouble a = mul_pwr2(addqq(spq, dpq), 0.5);
    *smin = absq(divqq(ha, a));
    *smax = absq(mulqq(fa, a));

    if (compute_uv) {
        ddouble tmp = addqq(divqq(q, addqq(spq, s)),
                            divqq(q, addqq(dpq, d)));
        tmp = mulqq(tmp, adddq(1.0, a));
        ddouble tt = hypotqd(tmp, 2.0);
        *cv = divdq(2.0, tt);
        *sv = divqq(tmp, tt);
        *cu = divqq(addqq(*cv, mulqq(*sv, q)), a);
        *su = divqq(mulqq(divqq(h, f), *sv), a);
    }
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
