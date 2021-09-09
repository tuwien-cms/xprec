#pragma once
#include "dd_arith.h"

ddouble normq(const ddouble *x, long nn, long sxn);

void givensq(ddouble f, ddouble g, ddouble *c, ddouble *s, ddouble *r);

ddouble householderq(const ddouble *x, ddouble *v, long nn, long sx, long sv);

void svd_tri2x2(ddouble f, ddouble g, ddouble h, ddouble *smin, ddouble *smax,
                ddouble *cv, ddouble *sv, ddouble *cu, ddouble *su);

void mul_givensq(long i1, long i2, ddouble c, ddouble s, long jj,
                 ddouble *A, long sai, long saj);

void golub_kahan_chaseq(ddouble *d, long sd, ddouble *e, long se, long ii,
                        ddouble shift, ddouble *rot);
