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
