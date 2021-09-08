#pragma once
#include "dd_arith.h"

ddouble normq(const ddouble *x, long nn, long sxn);

void givensq(ddouble f, ddouble g, ddouble *c, ddouble *s, ddouble *r);

void svd_tri2x2(ddouble f, ddouble g, ddouble h, ddouble *smin, ddouble *smax,
                ddouble *cv, ddouble *sv, ddouble *cu, ddouble *su);
