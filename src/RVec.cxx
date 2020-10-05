#include "RVec.hxx"

#if (_VECOPS_USE_EXTERN_TEMPLATES)

namespace ROOT {
namespace VecOps {

#define RVEC_DECLARE_UNARY_OPERATOR(T, OP) template RVec<T> operator OP(const RVec<T> &);

#define RVEC_DECLARE_BINARY_OPERATOR(T, OP)                                            \
   template auto operator OP(const RVec<T> &v, const T &y)->RVec<decltype(v[0] OP y)>; \
   template auto operator OP(const T &x, const RVec<T> &v)->RVec<decltype(x OP v[0])>; \
   template auto operator OP(const RVec<T> &v0, const RVec<T> &v1)->RVec<decltype(v0[0] OP v1[0])>;

#define RVEC_DECLARE_LOGICAL_OPERATOR(T, OP)                   \
   template RVec<int> operator OP(const RVec<T> &, const T &); \
   template RVec<int> operator OP(const T &, const RVec<T> &); \
   template RVec<int> operator OP(const RVec<T> &, const RVec<T> &);

#define RVEC_DECLARE_ASSIGN_OPERATOR(T, OP)             \
   template RVec<T> &operator OP(RVec<T> &, const T &); \
   template RVec<T> &operator OP(RVec<T> &, const RVec<T> &);

#define RVEC_DECLARE_FLOAT_TEMPLATE(T)  \
   template class RVec<T>;              \
   RVEC_DECLARE_UNARY_OPERATOR(T, +)    \
   RVEC_DECLARE_UNARY_OPERATOR(T, -)    \
   RVEC_DECLARE_UNARY_OPERATOR(T, !)    \
   RVEC_DECLARE_BINARY_OPERATOR(T, +)   \
   RVEC_DECLARE_BINARY_OPERATOR(T, -)   \
   RVEC_DECLARE_BINARY_OPERATOR(T, *)   \
   RVEC_DECLARE_BINARY_OPERATOR(T, /)   \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, +=)  \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, -=)  \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, *=)  \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, /=)  \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, <)  \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, >)  \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, ==) \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, !=) \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, <=) \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, >=) \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, &&) \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, ||)

#define RVEC_DECLARE_INTEGER_TEMPLATE(T) \
   template class RVec<T>;               \
   RVEC_DECLARE_UNARY_OPERATOR(T, +)     \
   RVEC_DECLARE_UNARY_OPERATOR(T, -)     \
   RVEC_DECLARE_UNARY_OPERATOR(T, ~)     \
   RVEC_DECLARE_UNARY_OPERATOR(T, !)     \
   RVEC_DECLARE_BINARY_OPERATOR(T, +)    \
   RVEC_DECLARE_BINARY_OPERATOR(T, -)    \
   RVEC_DECLARE_BINARY_OPERATOR(T, *)    \
   RVEC_DECLARE_BINARY_OPERATOR(T, /)    \
   RVEC_DECLARE_BINARY_OPERATOR(T, %)    \
   RVEC_DECLARE_BINARY_OPERATOR(T, &)    \
   RVEC_DECLARE_BINARY_OPERATOR(T, |)    \
   RVEC_DECLARE_BINARY_OPERATOR(T, ^)    \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, +=)   \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, -=)   \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, *=)   \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, /=)   \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, %=)   \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, &=)   \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, |=)   \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, ^=)   \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, >>=)  \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, <<=)  \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, <)   \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, >)   \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, ==)  \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, !=)  \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, <=)  \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, >=)  \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, &&)  \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, ||)

RVEC_DECLARE_INTEGER_TEMPLATE(char)
RVEC_DECLARE_INTEGER_TEMPLATE(short)
RVEC_DECLARE_INTEGER_TEMPLATE(int)
RVEC_DECLARE_INTEGER_TEMPLATE(long)
RVEC_DECLARE_INTEGER_TEMPLATE(long long)

RVEC_DECLARE_INTEGER_TEMPLATE(unsigned char)
RVEC_DECLARE_INTEGER_TEMPLATE(unsigned short)
RVEC_DECLARE_INTEGER_TEMPLATE(unsigned int)
RVEC_DECLARE_INTEGER_TEMPLATE(unsigned long)
RVEC_DECLARE_INTEGER_TEMPLATE(unsigned long long)

RVEC_DECLARE_FLOAT_TEMPLATE(float)
RVEC_DECLARE_FLOAT_TEMPLATE(double)

#define RVEC_DECLARE_UNARY_FUNCTION(T, NAME, FUNC) template RVec<PromoteType<T>> NAME(const RVec<T> &);

#define RVEC_DECLARE_STD_UNARY_FUNCTION(T, F) RVEC_DECLARE_UNARY_FUNCTION(T, F, ::std::F)

#define RVEC_DECLARE_BINARY_FUNCTION(T0, T1, NAME, FUNC)                     \
   template RVec<PromoteTypes<T0, T1>> NAME(const RVec<T0> &v, const T1 &y); \
   template RVec<PromoteTypes<T0, T1>> NAME(const T0 &x, const RVec<T1> &v); \
   template RVec<PromoteTypes<T0, T1>> NAME(const RVec<T0> &v0, const RVec<T1> &v1);

#define RVEC_DECLARE_STD_BINARY_FUNCTION(T, F) RVEC_DECLARE_BINARY_FUNCTION(T, T, F, ::std::F)

#define RVEC_DECLARE_STD_FUNCTIONS(T)             \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, abs)        \
   RVEC_DECLARE_STD_BINARY_FUNCTION(T, fdim)      \
   RVEC_DECLARE_STD_BINARY_FUNCTION(T, fmod)      \
   RVEC_DECLARE_STD_BINARY_FUNCTION(T, remainder) \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, exp)        \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, exp2)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, expm1)      \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, log)        \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, log10)      \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, log2)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, log1p)      \
   RVEC_DECLARE_STD_BINARY_FUNCTION(T, pow)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, sqrt)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, cbrt)       \
   RVEC_DECLARE_STD_BINARY_FUNCTION(T, hypot)     \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, sin)        \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, cos)        \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, tan)        \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, asin)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, acos)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, atan)       \
   RVEC_DECLARE_STD_BINARY_FUNCTION(T, atan2)     \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, sinh)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, cosh)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, tanh)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, asinh)      \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, acosh)      \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, atanh)      \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, floor)      \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, ceil)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, trunc)      \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, round)      \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, lround)     \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, llround)    \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, erf)        \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, erfc)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, lgamma)     \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, tgamma)

RVEC_DECLARE_STD_FUNCTIONS(float)
RVEC_DECLARE_STD_FUNCTIONS(double)
#undef RVEC_DECLARE_STD_UNARY_FUNCTION
#undef RVEC_DECLARE_STD_BINARY_FUNCTION
#undef RVEC_DECLARE_STD_UNARY_FUNCTIONS

#ifdef R__HAS_VDT

#define RVEC_DECLARE_VDT_UNARY_FUNCTION(T, F) RVEC_DECLARE_UNARY_FUNCTION(T, F, vdt::F)

RVEC_DECLARE_VDT_UNARY_FUNCTION(float, fast_expf)
RVEC_DECLARE_VDT_UNARY_FUNCTION(float, fast_logf)
RVEC_DECLARE_VDT_UNARY_FUNCTION(float, fast_sinf)
RVEC_DECLARE_VDT_UNARY_FUNCTION(float, fast_cosf)
RVEC_DECLARE_VDT_UNARY_FUNCTION(float, fast_tanf)
RVEC_DECLARE_VDT_UNARY_FUNCTION(float, fast_asinf)
RVEC_DECLARE_VDT_UNARY_FUNCTION(float, fast_acosf)
RVEC_DECLARE_VDT_UNARY_FUNCTION(float, fast_atanf)

RVEC_DECLARE_VDT_UNARY_FUNCTION(double, fast_exp)
RVEC_DECLARE_VDT_UNARY_FUNCTION(double, fast_log)
RVEC_DECLARE_VDT_UNARY_FUNCTION(double, fast_sin)
RVEC_DECLARE_VDT_UNARY_FUNCTION(double, fast_cos)
RVEC_DECLARE_VDT_UNARY_FUNCTION(double, fast_tan)
RVEC_DECLARE_VDT_UNARY_FUNCTION(double, fast_asin)
RVEC_DECLARE_VDT_UNARY_FUNCTION(double, fast_acos)
RVEC_DECLARE_VDT_UNARY_FUNCTION(double, fast_atan)

#endif // R__HAS_VDT

} // namespace VecOps
} // namespace ROOT

#endif // _VECOPS_USE_EXTERN_TEMPLATES
