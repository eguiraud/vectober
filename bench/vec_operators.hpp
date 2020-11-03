// Copyright (C) 2020 Enrico Guiraud (see LICENSE file in top directory)
// Author: Enrico Guiraud

#include <algorithm> // std::transform
#include <cmath>     // pow
#include <stdexcept> // std::runtime_error

// Binary Arithmetic Operators for any vector type

#define ERROR_MESSAGE(OP) "Cannot call operator " #OP " on vectors of different sizes."

#define VEC_BINARY_OPERATOR(OP)                                                                                  \
   template <typename Vec_t, typename T>                                                                         \
   auto operator OP(const Vec_t &v, const T &y)->decltype(typename Vec_t::value_type{}, Vec_t{})                 \
   {                                                                                                             \
      Vec_t ret(v.size());                                                                                       \
      auto op = [&y](const T &x) { return x OP y; };                                                             \
      std::transform(v.begin(), v.end(), ret.begin(), op);                                                       \
      return ret;                                                                                                \
   }                                                                                                             \
                                                                                                                 \
   template <typename T0, typename Vec_t>                                                                        \
   auto operator OP(const T0 &x, const Vec_t &v)->decltype(typename Vec_t::value_type{}, Vec_t{})                \
   {                                                                                                             \
      Vec_t ret(v.size());                                                                                       \
      auto op = [&x](const typename Vec_t::value_type &y) { return x OP y; };                                    \
      std::transform(v.begin(), v.end(), ret.begin(), op);                                                       \
      return ret;                                                                                                \
   }                                                                                                             \
                                                                                                                 \
   template <typename Vec_t>                                                                                     \
   auto operator OP(const Vec_t &v0, const Vec_t &v1)->decltype(typename Vec_t::value_type{}, Vec_t{})           \
   {                                                                                                             \
      if (v0.size() != v1.size())                                                                                \
         throw std::runtime_error(ERROR_MESSAGE(OP));                                                            \
                                                                                                                 \
      Vec_t ret(v0.size());                                                                                      \
      auto op = [](const typename Vec_t::value_type &x, const typename Vec_t::value_type &y) { return x OP y; }; \
      std::transform(v0.begin(), v0.end(), v1.begin(), ret.begin(), op);                                         \
      return ret;                                                                                                \
   }

VEC_BINARY_OPERATOR(+)
VEC_BINARY_OPERATOR(-)
VEC_BINARY_OPERATOR(*)
VEC_BINARY_OPERATOR(/)
VEC_BINARY_OPERATOR(%)
VEC_BINARY_OPERATOR(^)
VEC_BINARY_OPERATOR(|)
VEC_BINARY_OPERATOR(&)
#undef VEC_BINARY_OPERATOR

// Comparison and Logical Operators for any vector type

#define VEC_LOGICAL_OPERATOR(OP)                                              \
template <template <typename, unsigned> class Vec_t, unsigned N, typename T1, typename T2>        \
auto operator OP(const Vec_t<T1, N> &v, const T2 &y)                               \
  -> Vec_t<int, N> /* avoid std::vector<bool> */                                   \
{                                                                              \
   Vec_t<int, N> ret(v.size());                                                    \
   auto op = [y](const T1 &x) -> int { return x OP y; };                       \
   std::transform(v.begin(), v.end(), ret.begin(), op);                        \
   return ret;                                                                 \
}                                                                              \
                                                                               \
template <template <typename, unsigned> class Vec_t, unsigned N, typename T0, typename T1>                                            \
auto operator OP(const T0 &x, const Vec_t<T1, N> &v)                               \
  -> Vec_t<int, N> /* avoid std::vector<bool> */                                   \
{                                                                              \
   Vec_t<int, N> ret(v.size());                                                    \
   auto op = [x](const T1 &y) -> int { return x OP y; };                       \
   std::transform(v.begin(), v.end(), ret.begin(), op);                        \
   return ret;                                                                 \
}                                                                              \
                                                                               \
template <template <typename, unsigned> class Vec_t, unsigned N1, unsigned N2, typename T0, typename T1>                                            \
auto operator OP(const Vec_t<T0, N1> &v0, const Vec_t<T1, N2> &v1)                       \
  -> Vec_t<int, N1> /* avoid std::vector<bool> */                                   \
{                                                                              \
   if (v0.size() != v1.size())                                                 \
      throw std::runtime_error(ERROR_MESSAGE(OP));                             \
                                                                               \
   Vec_t<int, N1> ret(v0.size());                                                   \
   auto op = [](const T0 &x, const T1 &y) -> int { return x OP y; };           \
   std::transform(v0.begin(), v0.end(), v1.begin(), ret.begin(), op);          \
   return ret;                                                                 \
}                                                                              \

VEC_LOGICAL_OPERATOR(<)
VEC_LOGICAL_OPERATOR(>)
VEC_LOGICAL_OPERATOR(==)
VEC_LOGICAL_OPERATOR(!=)
VEC_LOGICAL_OPERATOR(<=)
VEC_LOGICAL_OPERATOR(>=)
VEC_LOGICAL_OPERATOR(&&)
VEC_LOGICAL_OPERATOR(||)
#undef VEC_LOGICAL_OPERATOR

///@}
