#include <fenv.h>

#include "dd_arith.h"

const int CATCHINVALID =
    FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW;

#define OPERQD_FULL(oper_full, oper, op)                        \
    ddouble oper_full(ddouble x, double y)                      \
    {                                                           \
        feclearexcept(CATCHINVALID);                            \
        ddouble res = oper(x, y);                               \
        if (fetestexcept(CATCHINVALID)) {                       \
            feclearexcept(CATCHINVALID);                        \
            res = (ddouble) {x.hi op y, 0.0};                   \
        }                                                       \
        return res;                                             \
    }

#define OPERDQ_FULL(oper_full, oper, op)                        \
    ddouble oper_full(double x, ddouble y)                      \
    {                                                           \
        feclearexcept(CATCHINVALID);                            \
        ddouble res = oper(x, y);                               \
        if (fetestexcept(CATCHINVALID)) {                       \
            feclearexcept(CATCHINVALID);                        \
            res = (ddouble) {x op y.hi, 0.0};                   \
        }                                                       \
        return res;                                             \
    }

#define OPERQQ_FULL(oper_full, oper, op)                        \
    ddouble oper_full(ddouble x, ddouble y)                     \
    {                                                           \
        feclearexcept(CATCHINVALID);                            \
        ddouble res = oper(x, y);                               \
        if (fetestexcept(CATCHINVALID)) {                       \
            feclearexcept(CATCHINVALID);                        \
            res = (ddouble) {x.hi op y.hi, 0.0};                \
        }                                                       \
        return res;                                             \
    }

OPERQD_FULL(addqd_full, addqd, +)
OPERQD_FULL(subqd_full, subqd, -)
OPERQD_FULL(mulqd_full, mulqd, *)
OPERQD_FULL(divqd_full, divqd, /)

OPERDQ_FULL(adddq_full, adddq, +)
OPERDQ_FULL(subdq_full, subdq, -)
OPERDQ_FULL(muldq_full, muldq, *)
OPERDQ_FULL(divdq_full, divdq, /)

OPERQQ_FULL(addqq_full, addqq, +)
OPERQQ_FULL(subqq_full, subqq, -)
OPERQQ_FULL(mulqq_full, mulqq, *)
OPERQQ_FULL(divqq_full, divqq, /)
