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
///@name Standard Mathematical Functions
///@{

/// \cond
template <typename T> struct PromoteTypeImpl;

template <> struct PromoteTypeImpl<float>       { using Type = float;       };
template <> struct PromoteTypeImpl<double>      { using Type = double;      };
template <> struct PromoteTypeImpl<long double> { using Type = long double; };

template <typename T> struct PromoteTypeImpl { using Type = double; };

template <typename T>
using PromoteType = typename PromoteTypeImpl<T>::Type;

template <typename U, typename V>
using PromoteTypes = decltype(PromoteType<U>() + PromoteType<V>());

/// \endcond

#define VEC_UNARY_FUNCTION(NAME, FUNC)                                        \
   template <template <typename, unsigned> class Vec_t, typename T, unsigned N>\
   Vec_t<PromoteType<T>, N> NAME(const Vec_t<T, N> &v)                                 \
   {                                                                           \
      Vec_t<PromoteType<T>, N> ret(v.size());                                      \
      auto f = [](const T &x) { return FUNC(x); };                             \
      std::transform(v.begin(), v.end(), ret.begin(), f);                      \
      return ret;                                                              \
   }

#define VEC_BINARY_FUNCTION(NAME, FUNC)                                       \
   template <typename T0, template <typename, unsigned> class Vec_t, typename T1, unsigned N>\
   Vec_t<PromoteTypes<T0, T1>, N> NAME(const T0 &x, const Vec_t<T1, N> &v)             \
   {                                                                           \
      Vec_t<PromoteTypes<T0, T1>, N> ret(v.size());                                \
      auto f = [&x](const T1 &y) { return FUNC(x, y); };                       \
      std::transform(v.begin(), v.end(), ret.begin(), f);                      \
      return ret;                                                              \
   }                                                                           \
                                                                               \
   template <typename T0, template <typename, unsigned> class Vec_t, typename T1, unsigned N>\
   Vec_t<PromoteTypes<T0, T1>, N> NAME(const Vec_t<T0, N> &v, const T1 &y)             \
   {                                                                           \
      Vec_t<PromoteTypes<T0, T1>, N> ret(v.size());                                \
      auto f = [&y](const T1 &x) { return FUNC(x, y); };                       \
      std::transform(v.begin(), v.end(), ret.begin(), f);                      \
      return ret;                                                              \
   }                                                                           \
                                                                               \
   template <template <typename, unsigned> class Vec_t, typename T0, unsigned N0, typename T1, unsigned N1>\
   Vec_t<PromoteTypes<T0, T1>, N1> NAME(const Vec_t<T0, N0> &v0, const Vec_t<T1, N1> &v1)     \
   {                                                                           \
      if (v0.size() != v1.size())                                              \
         throw std::runtime_error(ERROR_MESSAGE(NAME));                        \
                                                                               \
      Vec_t<PromoteTypes<T0, T1>, N1> ret(v0.size());                               \
      auto f = [](const T0 &x, const T1 &y) { return FUNC(x, y); };            \
      std::transform(v0.begin(), v0.end(), v1.begin(), ret.begin(), f);        \
      return ret;                                                              \
   }                                                                           \

#define VEC_STD_UNARY_FUNCTION(F) VEC_UNARY_FUNCTION(F, std::F)
#define VEC_STD_BINARY_FUNCTION(F) VEC_BINARY_FUNCTION(F, std::F)

VEC_STD_UNARY_FUNCTION(abs)
VEC_STD_BINARY_FUNCTION(fdim)
VEC_STD_BINARY_FUNCTION(fmod)
VEC_STD_BINARY_FUNCTION(remainder)

VEC_STD_UNARY_FUNCTION(exp)
VEC_STD_UNARY_FUNCTION(exp2)
VEC_STD_UNARY_FUNCTION(expm1)

VEC_STD_UNARY_FUNCTION(log)
VEC_STD_UNARY_FUNCTION(log10)
VEC_STD_UNARY_FUNCTION(log2)
VEC_STD_UNARY_FUNCTION(log1p)

VEC_STD_BINARY_FUNCTION(pow)
VEC_STD_UNARY_FUNCTION(sqrt)
VEC_STD_UNARY_FUNCTION(cbrt)
VEC_STD_BINARY_FUNCTION(hypot)

VEC_STD_UNARY_FUNCTION(sin)
VEC_STD_UNARY_FUNCTION(cos)
VEC_STD_UNARY_FUNCTION(tan)
VEC_STD_UNARY_FUNCTION(asin)
VEC_STD_UNARY_FUNCTION(acos)
VEC_STD_UNARY_FUNCTION(atan)
VEC_STD_BINARY_FUNCTION(atan2)

VEC_STD_UNARY_FUNCTION(sinh)
VEC_STD_UNARY_FUNCTION(cosh)
VEC_STD_UNARY_FUNCTION(tanh)
VEC_STD_UNARY_FUNCTION(asinh)
VEC_STD_UNARY_FUNCTION(acosh)
VEC_STD_UNARY_FUNCTION(atanh)

VEC_STD_UNARY_FUNCTION(floor)
VEC_STD_UNARY_FUNCTION(ceil)
VEC_STD_UNARY_FUNCTION(trunc)
VEC_STD_UNARY_FUNCTION(round)
VEC_STD_UNARY_FUNCTION(lround)
VEC_STD_UNARY_FUNCTION(llround)

VEC_STD_UNARY_FUNCTION(erf)
VEC_STD_UNARY_FUNCTION(erfc)
VEC_STD_UNARY_FUNCTION(lgamma)
VEC_STD_UNARY_FUNCTION(tgamma)
#undef VEC_STD_UNARY_FUNCTION

template <typename T>
T DeltaPhi(T v1, T v2, const T c = M_PI)
{
   static_assert(std::is_floating_point<T>::value, "DeltaPhi must be called with floating point values.");
   auto r = std::fmod(v2 - v1, 2.0 * c);
   if (r < -c) {
      r += 2.0 * c;
   } else if (r > c) {
      r -= 2.0 * c;
   }
   return r;
}

template <typename Vec_t, typename T>
auto DeltaPhi(const Vec_t &v1, const Vec_t &v2, const T c = M_PI) -> decltype(typename Vec_t::value_type{}, Vec_t{})
{
   using size_type = typename Vec_t::size_type;
   const size_type size = v1.size();
   auto r = Vec_t(size);
   for (size_type i = 0; i < size; i++) {
      r[i] = DeltaPhi(v1[i], v2[i], c);
   }
   return r;
}

template <typename Vec_t, typename T>
Vec_t DeltaPhi(const Vec_t &v1, T v2, const T c = M_PI)
{
   using size_type = typename Vec_t::size_type;
   const size_type size = v1.size();
   auto r = Vec_t(size);
   for (size_type i = 0; i < size; i++) {
      r[i] = DeltaPhi(v1[i], v2, c);
   }
   return r;
}

template <typename Vec_t, typename T>
Vec_t DeltaPhi(T v1, const Vec_t &v2, const T c = M_PI)
{
   using size_type = typename Vec_t::size_type;
   const size_type size = v2.size();
   auto r = Vec_t(size);
   for (size_type i = 0; i < size; i++) {
      r[i] = DeltaPhi(v1, v2[i], c);
   }
   return r;
}

template <typename Vec_t>
Vec_t DeltaR2(const Vec_t &eta1, const Vec_t &eta2, const Vec_t &phi1, const Vec_t &phi2, const typename Vec_t::value_type c = M_PI)
{
   const auto dphi = DeltaPhi(phi1, phi2, c);
   return (eta1 - eta2) * (eta1 - eta2) + dphi * dphi;
}

template <typename Vec_t>
Vec_t DeltaR(const Vec_t &eta1, const Vec_t &eta2, const Vec_t &phi1, const Vec_t &phi2, const typename Vec_t::value_type c = M_PI)
{
   return sqrt(DeltaR2(eta1, eta2, phi1, phi2, c));
}

/* don't need this one for our benchmarks
template <typename T>
T DeltaR(T eta1, T eta2, T phi1, T phi2, const T c = M_PI)
{
   const auto dphi = DeltaPhi(phi1, phi2, c);
   return std::sqrt((eta1 - eta2) * (eta1 - eta2) + dphi * dphi);
}
*/

template <typename Vec_t>
auto InvariantMasses(const Vec_t &pt1, const Vec_t &eta1, const Vec_t &phi1, const Vec_t &mass1,
                        const Vec_t &pt2, const Vec_t &eta2, const Vec_t &phi2, const Vec_t &mass2) -> decltype(typename Vec_t::value_type{}, Vec_t{})
{
   std::size_t size = pt1.size();

   assert(eta1.size() == size && phi1.size() == size && mass1.size() == size);
   assert(pt2.size() == size && phi2.size() == size && mass2.size() == size);

   Vec_t inv_masses(size);

   for (std::size_t i = 0u; i < size; ++i) {
      // Conversion from (pt, eta, phi, mass) to (x, y, z, e) coordinate system
      const auto x1 = pt1[i] * std::cos(phi1[i]);
      const auto y1 = pt1[i] * std::sin(phi1[i]);
      const auto z1 = pt1[i] * std::sinh(eta1[i]);
      const auto e1 = std::sqrt(x1 * x1 + y1 * y1 + z1 * z1 + mass1[i] * mass1[i]);

      const auto x2 = pt2[i] * std::cos(phi2[i]);
      const auto y2 = pt2[i] * std::sin(phi2[i]);
      const auto z2 = pt2[i] * std::sinh(eta2[i]);
      const auto e2 = std::sqrt(x2 * x2 + y2 * y2 + z2 * z2 + mass2[i] * mass2[i]);

      // Addition of particle four-vector elements
      const auto e = e1 + e2;
      const auto x = x1 + x2;
      const auto y = y1 + y2;
      const auto z = z1 + z2;

      inv_masses[i] = std::sqrt(e * e - x * x - y * y - z * z);
   }

   // Return invariant mass with (+, -, -, -) metric
   return inv_masses;
}

template <typename Vec_t, typename T = typename Vec_t::value_type>
T InvariantMass(const Vec_t &pt, const Vec_t &eta, const Vec_t &phi, const Vec_t &mass)
{
   const std::size_t size = pt.size();

   assert(eta.size() == size && phi.size() == size && mass.size() == size);

   T x_sum = 0.;
   T y_sum = 0.;
   T z_sum = 0.;
   T e_sum = 0.;

   for (std::size_t i = 0u; i < size; ++i) {
      // Convert to (e, x, y, z) coordinate system and update sums
      const auto x = pt[i] * std::cos(phi[i]);
      x_sum += x;
      const auto y = pt[i] * std::sin(phi[i]);
      y_sum += y;
      const auto z = pt[i] * std::sinh(eta[i]);
      z_sum += z;
      const auto e = std::sqrt(x * x + y * y + z * z + mass[i] * mass[i]);
      e_sum += e;
   }

   // Return invariant mass with (+, -, -, -) metric
   return std::sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum - z_sum * z_sum);
}
