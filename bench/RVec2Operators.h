#include <cmath>
#include <algorithm>
#include <RIntegerSequence.hxx>
#include <stdexcept> // std::runtime_error
#include <cassert>
#include <sstream>
#include <numeric>
#include <string>

#define R__ASSERT assert

namespace ROOT {

template <typename... T>
std::size_t GetVectorsSize(std::string id, const RVec<T> &... vs)
{
   constexpr const auto nArgs = sizeof...(T);
   const std::size_t sizes[] = {vs.size()...};
   if (nArgs > 1) {
      for (auto i = 1UL; i < nArgs; i++) {
         if (sizes[0] == sizes[i])
            continue;
         std::string msg(id);
         msg += ": input RVec instances have different lengths!";
         throw std::runtime_error(msg);
      }
   }
   return sizes[0];
}

template <typename F, typename... T>
auto MapImpl(F &&f, const RVec<T> &... vs) -> RVec<decltype(f(vs[0]...))>
{
   const auto size = GetVectorsSize("Map", vs...);
   RVec<decltype(f(vs[0]...))> ret(size);

   for (auto i = 0UL; i < size; i++)
      ret[i] = f(vs[i]...);

   return ret;
}

template <typename Tuple_t, std::size_t... Is>
auto MapFromTuple(Tuple_t &&t, std::index_sequence<Is...>)
   -> decltype(MapImpl(std::get<std::tuple_size<Tuple_t>::value - 1>(t), std::get<Is>(t)...))
{
   constexpr const auto tupleSizeM1 = std::tuple_size<Tuple_t>::value - 1;
   return MapImpl(std::get<tupleSizeM1>(t), std::get<Is>(t)...);
}

///@name RVec Unary Arithmetic Operators
///@{

#define RVEC_UNARY_OPERATOR(OP)                                                \
template <typename T>                                                          \
RVec<T> operator OP(const RVec<T> &v)                                          \
{                                                                              \
   RVec<T> ret(v);                                                             \
   for (auto &x : ret)                                                         \
      x = OP x;                                                                \
return ret;                                                                    \
}                                                                              \

RVEC_UNARY_OPERATOR(+)
RVEC_UNARY_OPERATOR(-)
RVEC_UNARY_OPERATOR(~)
RVEC_UNARY_OPERATOR(!)
#undef RVEC_UNARY_OPERATOR

///@}
///@name RVec Binary Arithmetic Operators
///@{

#define ERROR_MESSAGE(OP) \
 "Cannot call operator " #OP " on vectors of different sizes."

#define RVEC_BINARY_OPERATOR(OP)                                               \
template <typename T0, typename T1>                                            \
auto operator OP(const RVec<T0> &v, const T1 &y)                               \
  -> RVec<decltype(v[0] OP y)>                                                 \
{                                                                              \
   RVec<decltype(v[0] OP y)> ret(v.size());                                    \
   auto op = [&y](const T0 &x) { return x OP y; };                             \
   std::transform(v.begin(), v.end(), ret.begin(), op);                        \
   return ret;                                                                 \
}                                                                              \
                                                                               \
template <typename T0, typename T1>                                            \
auto operator OP(const T0 &x, const RVec<T1> &v)                               \
  -> RVec<decltype(x OP v[0])>                                                 \
{                                                                              \
   RVec<decltype(x OP v[0])> ret(v.size());                                    \
   auto op = [&x](const T1 &y) { return x OP y; };                             \
   std::transform(v.begin(), v.end(), ret.begin(), op);                        \
   return ret;                                                                 \
}                                                                              \
                                                                               \
template <typename T0, typename T1>                                            \
auto operator OP(const RVec<T0> &v0, const RVec<T1> &v1)                       \
  -> RVec<decltype(v0[0] OP v1[0])>                                            \
{                                                                              \
   if (v0.size() != v1.size())                                                 \
      throw std::runtime_error(ERROR_MESSAGE(OP));                             \
                                                                               \
   RVec<decltype(v0[0] OP v1[0])> ret(v0.size());                              \
   auto op = [](const T0 &x, const T1 &y) { return x OP y; };                  \
   std::transform(v0.begin(), v0.end(), v1.begin(), ret.begin(), op);          \
   return ret;                                                                 \
}                                                                              \

RVEC_BINARY_OPERATOR(+)
RVEC_BINARY_OPERATOR(-)
RVEC_BINARY_OPERATOR(*)
RVEC_BINARY_OPERATOR(/)
RVEC_BINARY_OPERATOR(%)
RVEC_BINARY_OPERATOR(^)
RVEC_BINARY_OPERATOR(|)
RVEC_BINARY_OPERATOR(&)
#undef RVEC_BINARY_OPERATOR

///@}
///@name RVec Assignment Arithmetic Operators
///@{

#define RVEC_ASSIGNMENT_OPERATOR(OP)                                           \
template <typename T0, typename T1>                                            \
RVec<T0>& operator OP(RVec<T0> &v, const T1 &y)                                \
{                                                                              \
   auto op = [&y](T0 &x) { return x OP y; };                                   \
   std::transform(v.begin(), v.end(), v.begin(), op);                          \
   return v;                                                                   \
}                                                                              \
                                                                               \
template <typename T0, typename T1>                                            \
RVec<T0>& operator OP(RVec<T0> &v0, const RVec<T1> &v1)                        \
{                                                                              \
   if (v0.size() != v1.size())                                                 \
      throw std::runtime_error(ERROR_MESSAGE(OP));                             \
                                                                               \
   auto op = [](T0 &x, const T1 &y) { return x OP y; };                        \
   std::transform(v0.begin(), v0.end(), v1.begin(), v0.begin(), op);           \
   return v0;                                                                  \
}                                                                              \

RVEC_ASSIGNMENT_OPERATOR(+=)
RVEC_ASSIGNMENT_OPERATOR(-=)
RVEC_ASSIGNMENT_OPERATOR(*=)
RVEC_ASSIGNMENT_OPERATOR(/=)
RVEC_ASSIGNMENT_OPERATOR(%=)
RVEC_ASSIGNMENT_OPERATOR(^=)
RVEC_ASSIGNMENT_OPERATOR(|=)
RVEC_ASSIGNMENT_OPERATOR(&=)
RVEC_ASSIGNMENT_OPERATOR(>>=)
RVEC_ASSIGNMENT_OPERATOR(<<=)
#undef RVEC_ASSIGNMENT_OPERATOR

///@}
///@name RVec Comparison and Logical Operators
///@{

#define RVEC_LOGICAL_OPERATOR(OP)                                              \
template <typename T0, typename T1>                                            \
auto operator OP(const RVec<T0> &v, const T1 &y)                               \
  -> RVec<int> /* avoid std::vector<bool> */                                   \
{                                                                              \
   RVec<int> ret(v.size());                                                    \
   auto op = [y](const T0 &x) -> int { return x OP y; };                       \
   std::transform(v.begin(), v.end(), ret.begin(), op);                        \
   return ret;                                                                 \
}                                                                              \
                                                                               \
template <typename T0, typename T1>                                            \
auto operator OP(const T0 &x, const RVec<T1> &v)                               \
  -> RVec<int> /* avoid std::vector<bool> */                                   \
{                                                                              \
   RVec<int> ret(v.size());                                                    \
   auto op = [x](const T1 &y) -> int { return x OP y; };                       \
   std::transform(v.begin(), v.end(), ret.begin(), op);                        \
   return ret;                                                                 \
}                                                                              \
                                                                               \
template <typename T0, typename T1>                                            \
auto operator OP(const RVec<T0> &v0, const RVec<T1> &v1)                       \
  -> RVec<int> /* avoid std::vector<bool> */                                   \
{                                                                              \
   if (v0.size() != v1.size())                                                 \
      throw std::runtime_error(ERROR_MESSAGE(OP));                             \
                                                                               \
   RVec<int> ret(v0.size());                                                   \
   auto op = [](const T0 &x, const T1 &y) -> int { return x OP y; };           \
   std::transform(v0.begin(), v0.end(), v1.begin(), ret.begin(), op);          \
   return ret;                                                                 \
}                                                                              \

RVEC_LOGICAL_OPERATOR(<)
RVEC_LOGICAL_OPERATOR(>)
RVEC_LOGICAL_OPERATOR(==)
RVEC_LOGICAL_OPERATOR(!=)
RVEC_LOGICAL_OPERATOR(<=)
RVEC_LOGICAL_OPERATOR(>=)
RVEC_LOGICAL_OPERATOR(&&)
RVEC_LOGICAL_OPERATOR(||)
#undef RVEC_LOGICAL_OPERATOR

///@}
///@name RVec Standard Mathematical Functions
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

#define RVEC_UNARY_FUNCTION(NAME, FUNC)                                        \
   template <typename T>                                                       \
   RVec<PromoteType<T>> NAME(const RVec<T> &v)                                 \
   {                                                                           \
      RVec<PromoteType<T>> ret(v.size());                                      \
      auto f = [](const T &x) { return FUNC(x); };                             \
      std::transform(v.begin(), v.end(), ret.begin(), f);                      \
      return ret;                                                              \
   }

#define RVEC_BINARY_FUNCTION(NAME, FUNC)                                       \
   template <typename T0, typename T1>                                         \
   RVec<PromoteTypes<T0, T1>> NAME(const T0 &x, const RVec<T1> &v)             \
   {                                                                           \
      RVec<PromoteTypes<T0, T1>> ret(v.size());                                \
      auto f = [&x](const T1 &y) { return FUNC(x, y); };                       \
      std::transform(v.begin(), v.end(), ret.begin(), f);                      \
      return ret;                                                              \
   }                                                                           \
                                                                               \
   template <typename T0, typename T1>                                         \
   RVec<PromoteTypes<T0, T1>> NAME(const RVec<T0> &v, const T1 &y)             \
   {                                                                           \
      RVec<PromoteTypes<T0, T1>> ret(v.size());                                \
      auto f = [&y](const T1 &x) { return FUNC(x, y); };                       \
      std::transform(v.begin(), v.end(), ret.begin(), f);                      \
      return ret;                                                              \
   }                                                                           \
                                                                               \
   template <typename T0, typename T1>                                         \
   RVec<PromoteTypes<T0, T1>> NAME(const RVec<T0> &v0, const RVec<T1> &v1)     \
   {                                                                           \
      if (v0.size() != v1.size())                                              \
         throw std::runtime_error(ERROR_MESSAGE(NAME));                        \
                                                                               \
      RVec<PromoteTypes<T0, T1>> ret(v0.size());                               \
      auto f = [](const T0 &x, const T1 &y) { return FUNC(x, y); };            \
      std::transform(v0.begin(), v0.end(), v1.begin(), ret.begin(), f);        \
      return ret;                                                              \
   }                                                                           \

#define RVEC_STD_UNARY_FUNCTION(F) RVEC_UNARY_FUNCTION(F, std::F)
#define RVEC_STD_BINARY_FUNCTION(F) RVEC_BINARY_FUNCTION(F, std::F)

RVEC_STD_UNARY_FUNCTION(abs)
RVEC_STD_BINARY_FUNCTION(fdim)
RVEC_STD_BINARY_FUNCTION(fmod)
RVEC_STD_BINARY_FUNCTION(remainder)

RVEC_STD_UNARY_FUNCTION(exp)
RVEC_STD_UNARY_FUNCTION(exp2)
RVEC_STD_UNARY_FUNCTION(expm1)

RVEC_STD_UNARY_FUNCTION(log)
RVEC_STD_UNARY_FUNCTION(log10)
RVEC_STD_UNARY_FUNCTION(log2)
RVEC_STD_UNARY_FUNCTION(log1p)

RVEC_STD_BINARY_FUNCTION(pow)
RVEC_STD_UNARY_FUNCTION(sqrt)
RVEC_STD_UNARY_FUNCTION(cbrt)
RVEC_STD_BINARY_FUNCTION(hypot)

RVEC_STD_UNARY_FUNCTION(sin)
RVEC_STD_UNARY_FUNCTION(cos)
RVEC_STD_UNARY_FUNCTION(tan)
RVEC_STD_UNARY_FUNCTION(asin)
RVEC_STD_UNARY_FUNCTION(acos)
RVEC_STD_UNARY_FUNCTION(atan)
RVEC_STD_BINARY_FUNCTION(atan2)

RVEC_STD_UNARY_FUNCTION(sinh)
RVEC_STD_UNARY_FUNCTION(cosh)
RVEC_STD_UNARY_FUNCTION(tanh)
RVEC_STD_UNARY_FUNCTION(asinh)
RVEC_STD_UNARY_FUNCTION(acosh)
RVEC_STD_UNARY_FUNCTION(atanh)

RVEC_STD_UNARY_FUNCTION(floor)
RVEC_STD_UNARY_FUNCTION(ceil)
RVEC_STD_UNARY_FUNCTION(trunc)
RVEC_STD_UNARY_FUNCTION(round)
RVEC_STD_UNARY_FUNCTION(lround)
RVEC_STD_UNARY_FUNCTION(llround)

RVEC_STD_UNARY_FUNCTION(erf)
RVEC_STD_UNARY_FUNCTION(erfc)
RVEC_STD_UNARY_FUNCTION(lgamma)
RVEC_STD_UNARY_FUNCTION(tgamma)
#undef RVEC_STD_UNARY_FUNCTION

///@}
///@name RVec Fast Mathematical Functions with Vdt
///@{

#ifdef R__HAS_VDT
#define RVEC_VDT_UNARY_FUNCTION(F) RVEC_UNARY_FUNCTION(F, vdt::F)

RVEC_VDT_UNARY_FUNCTION(fast_expf)
RVEC_VDT_UNARY_FUNCTION(fast_logf)
RVEC_VDT_UNARY_FUNCTION(fast_sinf)
RVEC_VDT_UNARY_FUNCTION(fast_cosf)
RVEC_VDT_UNARY_FUNCTION(fast_tanf)
RVEC_VDT_UNARY_FUNCTION(fast_asinf)
RVEC_VDT_UNARY_FUNCTION(fast_acosf)
RVEC_VDT_UNARY_FUNCTION(fast_atanf)

RVEC_VDT_UNARY_FUNCTION(fast_exp)
RVEC_VDT_UNARY_FUNCTION(fast_log)
RVEC_VDT_UNARY_FUNCTION(fast_sin)
RVEC_VDT_UNARY_FUNCTION(fast_cos)
RVEC_VDT_UNARY_FUNCTION(fast_tan)
RVEC_VDT_UNARY_FUNCTION(fast_asin)
RVEC_VDT_UNARY_FUNCTION(fast_acos)
RVEC_VDT_UNARY_FUNCTION(fast_atan)
#undef RVEC_VDT_UNARY_FUNCTION

#endif // R__HAS_VDT

#undef RVEC_UNARY_FUNCTION

///@}

/// Inner product
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<float> v1 {1., 2., 3.};
/// RVec<float> v2 {4., 5., 6.};
/// auto v1_dot_v2 = Dot(v1, v2);
/// v1_dot_v2
/// // (float) 32.f
/// ~~~
template <typename T, typename V>
auto Dot(const RVec<T> &v0, const RVec<V> &v1) -> decltype(v0[0] * v1[0])
{
   if (v0.size() != v1.size())
      throw std::runtime_error("Cannot compute inner product of vectors of different sizes");
   return std::inner_product(v0.begin(), v0.end(), v1.begin(), decltype(v0[0] * v1[0])(0));
}

/// Sum elements of an RVec
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<float> v {1.f, 2.f, 3.f};
/// auto v_sum = Sum(v);
/// v_sum
/// // (float) 6.f
/// ~~~
template <typename T>
T Sum(const RVec<T> &v)
{
   return std::accumulate(v.begin(), v.end(), T(0));
}

/// Get the mean of the elements of an RVec
///
/// The return type is a double precision floating point number.
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<float> v {1.f, 2.f, 4.f};
/// auto v_mean = Mean(v);
/// v_mean
/// // (double) 2.3333333
/// ~~~
template <typename T>
double Mean(const RVec<T> &v)
{
   if (v.empty()) return 0.;
   return double(Sum(v)) / v.size();
}

/// Get the greatest element of an RVec
///
/// Example code, at the ROOT prompt:
/// ~~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<float> v {1.f, 2.f, 4.f};
/// auto v_max = Max(v)
/// v_max
/// (float) 4.f
/// ~~~~
template <typename T>
T Max(const RVec<T> &v)
{
   return *std::max_element(v.begin(), v.end());
}

/// Get the smallest element of an RVec
///
/// Example code, at the ROOT prompt:
/// ~~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<float> v {1.f, 2.f, 4.f};
/// auto v_min = Min(v)
/// v_min
/// (float) 1.f
/// ~~~~
template <typename T>
T Min(const RVec<T> &v)
{
   return *std::min_element(v.begin(), v.end());
}

/// Get the index of the greatest element of an RVec
/// In case of multiple occurrences of the maximum values,
/// the index corresponding to the first occurrence is returned.
///
/// Example code, at the ROOT prompt:
/// ~~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<float> v {1.f, 2.f, 4.f};
/// auto v_argmax = ArgMax(v);
/// v_argmax
/// // (int) 2
/// ~~~~
template <typename T>
std::size_t ArgMax(const RVec<T> &v)
{
   return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

/// Get the index of the smallest element of an RVec
/// In case of multiple occurrences of the minimum values,
/// the index corresponding to the first occurrence is returned.
///
/// Example code, at the ROOT prompt:
/// ~~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<float> v {1.f, 2.f, 4.f};
/// auto v_argmin = ArgMin(v);
/// v_argmin
/// // (int) 0
/// ~~~~
template <typename T>
std::size_t ArgMin(const RVec<T> &v)
{
   return std::distance(v.begin(), std::min_element(v.begin(), v.end()));
}

/// Get the variance of the elements of an RVec
///
/// The return type is a double precision floating point number.
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<float> v {1.f, 2.f, 4.f};
/// auto v_var = Var(v);
/// v_var
/// // (double) 2.3333333
/// ~~~
template <typename T>
double Var(const RVec<T> &v)
{
   const std::size_t size = v.size();
   if (size < std::size_t(2)) return 0.;
   T sum_squares(0), squared_sum(0);
   auto pred = [&sum_squares, &squared_sum](const T& x) {sum_squares+=x*x; squared_sum+=x;};
   std::for_each(v.begin(), v.end(), pred);
   squared_sum *= squared_sum;
   const auto dsize = (double) size;
   return 1. / (dsize - 1.) * (sum_squares - squared_sum / dsize );
}

/// Get the standard deviation of the elements of an RVec
///
/// The return type is a double precision floating point number.
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<float> v {1.f, 2.f, 4.f};
/// auto v_sd = StdDev(v);
/// v_sd
/// // (double) 1.5275252
/// ~~~
template <typename T>
double StdDev(const RVec<T> &v)
{
   return std::sqrt(Var(v));
}

/// Create new collection applying a callable to the elements of the input collection
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<float> v {1.f, 2.f, 4.f};
/// auto v_square = Map(v, [](float f){return f* 2.f;});
/// v_square
/// // (ROOT::VecOps::RVec<float> &) { 2.00000f, 4.00000f, 8.00000f }
///
/// RVec<float> x({1.f, 2.f, 3.f});
/// RVec<float> y({4.f, 5.f, 6.f});
/// RVec<float> z({7.f, 8.f, 9.f});
/// auto mod = [](float x, float y, float z) { return sqrt(x * x + y * y + z * z); };
/// auto v_mod = Map(x, y, z, mod);
/// v_mod
/// // (ROOT::VecOps::RVec<float> &) { 8.12404f, 9.64365f, 11.2250f }
/// ~~~
template <typename... Args>
auto Map(Args &&... args)
   -> decltype(ROOT::MapFromTuple(std::forward_as_tuple(args...), std::make_index_sequence<sizeof...(args) - 1>()))
{
   /*
   Here the strategy in order to generalise the previous implementation of Map, i.e.
   `RVec Map(RVec, F)`, here we need to move the last parameter of the pack in first
   position in order to be able to invoke the Map function with automatic type deduction.
   This is achieved in two steps:
   1. Forward as tuple the pack to MapFromTuple
   2. Invoke the MapImpl helper which has the signature `template<...T, F> RVec MapImpl(F &&f, RVec<T>...)`
   NOTA BENE: the signature is very heavy but it is one of the lightest ways to manage in C++11
   to build the return type based on the template args.
   */
   return ROOT::MapFromTuple(std::forward_as_tuple(args...), std::make_index_sequence<sizeof...(args) - 1>());
}

/// Create a new collection with the elements passing the filter expressed by the predicate
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<int> v {1, 2, 4};
/// auto v_even = Filter(v, [](int i){return 0 == i%2;});
/// v_even
/// // (ROOT::VecOps::RVec<int> &) { 2, 4 }
/// ~~~
template <typename T, typename F>
RVec<T> Filter(const RVec<T> &v, F &&f)
{
   const auto thisSize = v.size();
   RVec<T> w;
   w.reserve(thisSize);
   for (auto &&val : v) {
      if (f(val))
         w.emplace_back(val);
   }
   return w;
}

/// Return true if any of the elements equates to true, return false otherwise.
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<int> v {0, 1, 0};
/// auto anyTrue = Any(v);
/// anyTrue
/// // (bool) true
/// ~~~
template <typename T>
auto Any(const RVec<T> &v) -> decltype(v[0] == true)
{
   for (auto &&e : v)
      if (e == true)
         return true;
   return false;
}

/// Return true if all of the elements equate to true, return false otherwise.
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<int> v {0, 1, 0};
/// auto allTrue = All(v);
/// allTrue
/// // (bool) false
/// ~~~
template <typename T>
auto All(const RVec<T> &v) -> decltype(v[0] == false)
{
   for (auto &&e : v)
      if (e == false)
         return false;
   return true;
}

template <typename T>
void swap(RVec<T> &lhs, RVec<T> &rhs)
{
   lhs.swap(rhs);
}

/// Return an RVec of indices that sort the input RVec
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<double> v {2., 3., 1.};
/// auto sortIndices = Argsort(v);
/// sortIndices
/// // (ROOT::VecOps::RVec<unsigned long> &) { 2, 0, 1 }
/// ~~~
template <typename T>
RVec<typename RVec<T>::size_type> Argsort(const RVec<T> &v)
{
   using size_type = typename RVec<T>::size_type;
   RVec<size_type> i(v.size());
   std::iota(i.begin(), i.end(), 0);
   std::sort(i.begin(), i.end(), [&v](size_type i1, size_type i2) { return v[i1] < v[i2]; });
   return i;
}

/// Return elements of a vector at given indices
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<double> v {2., 3., 1.};
/// auto vTaken = Take(v, {0,2});
/// vTaken
/// // (ROOT::VecOps::RVec<double>) { 2.0000000, 1.0000000 }
/// ~~~
template <typename T>
RVec<T> Take(const RVec<T> &v, const RVec<typename RVec<T>::size_type> &i)
{
   using size_type = typename RVec<T>::size_type;
   const size_type isize = i.size();
   RVec<T> r(isize);
   for (size_type k = 0; k < isize; k++)
      r[k] = v[i[k]];
   return r;
}

/// Return first or last `n` elements of an RVec
///
/// if `n > 0` and last elements if `n < 0`.
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<double> v {2., 3., 1.};
/// auto firstTwo = Take(v, 2);
/// firstTwo
/// // (ROOT::VecOps::RVec<double>) { 2.0000000, 3.0000000 }
/// auto lastOne = Take(v, -1);
/// lastOne
/// // (ROOT::VecOps::RVec<double>) { 1.0000000 }
/// ~~~
template <typename T>
RVec<T> Take(const RVec<T> &v, const int n)
{
   using size_type = typename RVec<T>::size_type;
   const size_type size = v.size();
   const size_type absn = std::abs(n);
   if (absn > size) {
      std::stringstream ss;
      ss << "Try to take " << absn << " elements but vector has only size " << size << ".";
      throw std::runtime_error(ss.str());
   }
   RVec<T> r(absn);
   if (n < 0) {
      for (size_type k = 0; k < absn; k++)
         r[k] = v[size - absn + k];
   } else {
      for (size_type k = 0; k < absn; k++)
         r[k] = v[k];
   }
   return r;
}

/// Return copy of reversed vector
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<double> v {2., 3., 1.};
/// auto v_reverse = Reverse(v);
/// v_reverse
/// // (ROOT::VecOps::RVec<double>) { 1.0000000, 3.0000000, 2.0000000 }
/// ~~~
template <typename T>
RVec<T> Reverse(const RVec<T> &v)
{
   RVec<T> r(v);
   std::reverse(r.begin(), r.end());
   return r;
}

/// Return copy of RVec with elements sorted in ascending order
///
/// This helper is different from ArgSort since it does not return an RVec of indices,
/// but an RVec of values.
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<double> v {2., 3., 1.};
/// auto v_sorted = Sort(v);
/// v_sorted
/// // (ROOT::VecOps::RVec<double>) { 1.0000000, 2.0000000, 3.0000000 }
/// ~~~
template <typename T>
RVec<T> Sort(const RVec<T> &v)
{
   RVec<T> r(v);
   std::sort(r.begin(), r.end());
   return r;
}

/// Return copy of RVec with elements sorted based on a comparison operator
///
/// The comparison operator has to fullfill the same requirements of the
/// predicate of by std::sort.
///
///
/// This helper is different from ArgSort since it does not return an RVec of indices,
/// but an RVec of values.
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<double> v {2., 3., 1.};
/// auto v_sorted = Sort(v, [](double x, double y) {return 1/x < 1/y;});
/// v_sorted
/// // (ROOT::VecOps::RVec<double>) { 3.0000000, 2.0000000, 1.0000000 }
/// ~~~
template <typename T, typename Compare>
RVec<T> Sort(const RVec<T> &v, Compare &&c)
{
   RVec<T> r(v);
   std::sort(r.begin(), r.end(), std::forward<Compare>(c));
   return r;
}

/// Return the indices that represent all combinations of the elements of two
/// RVecs.
///
/// The type of the return value is an RVec of two RVecs containing indices.
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// auto comb_idx = Combinations(3, 2);
/// comb_idx
/// // (ROOT::VecOps::RVec<ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>::size_type> >) { { 0, 0, 1, 1, 2, 2 }, { 0, 1, 0, 1, 0, 1 } }
/// ~~~
inline RVec<RVec<std::size_t>> Combinations(const std::size_t size1, const std::size_t size2)
{
   using size_type = std::size_t;
   RVec<RVec<size_type>> r(2);
   r[0].resize(size1*size2);
   r[1].resize(size1*size2);
   size_type c = 0;
   for(size_type i=0; i<size1; i++) {
      for(size_type j=0; j<size2; j++) {
         r[0][c] = i;
         r[1][c] = j;
         c++;
      }
   }
   return r;
}

/// Return the indices that represent all combinations of the elements of two
/// RVecs.
///
/// The type of the return value is an RVec of two RVecs containing indices.
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<double> v1 {1., 2., 3.};
/// RVec<double> v2 {-4., -5.};
/// auto comb_idx = Combinations(v1, v2);
/// comb_idx
/// // (ROOT::VecOps::RVec<ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>::size_type> >) { { 0, 0, 1, 1, 2, 2 }, { 0, 1,
/// 0, 1, 0, 1 } }
/// ~~~
template <typename T1, typename T2>
RVec<RVec<typename RVec<T1>::size_type>> Combinations(const RVec<T1> &v1, const RVec<T2> &v2)
{
   return Combinations(v1.size(), v2.size());
}

/// Return the indices that represent all unique combinations of the
/// elements of a given RVec.
///
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<double> v {1., 2., 3., 4.};
/// auto v_1 = Combinations(v, 1);
/// v_1
/// (ROOT::VecOps::RVec<ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>::size_type> >) { { 0, 1, 2, 3 } }
/// auto v_2 = Combinations(v, 2);
/// auto v_2
/// (ROOT::VecOps::RVec<ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>::size_type> >) { { 0, 0, 0, 1, 1, 2 }, { 1, 2, 3, 2, 3, 3 } }
/// auto v_3 = Combinations(v, 3);
/// v_3
/// (ROOT::VecOps::RVec<ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>::size_type> >) { { 0, 0, 0, 1 }, { 1, 1, 2, 2 }, { 2, 3, 3, 3 } }
/// auto v_4 = Combinations(v, 4);
/// v_4
/// (ROOT::VecOps::RVec<ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>::size_type> >) { { 0 }, { 1 }, { 2 }, { 3 } }
/// ~~~
template <typename T>
RVec<RVec<typename RVec<T>::size_type>> Combinations(const RVec<T>& v, const typename RVec<T>::size_type n)
{
   using size_type = typename RVec<T>::size_type;
   const size_type s = v.size();
   if (n > s) {
      std::stringstream ss;
      ss << "Cannot make unique combinations of size " << n << " from vector of size " << s << ".";
      throw std::runtime_error(ss.str());
   }
   RVec<size_type> indices(s);
   for(size_type k=0; k<s; k++)
      indices[k] = k;
   RVec<RVec<size_type>> c(n);
   for(size_type k=0; k<n; k++)
      c[k].emplace_back(indices[k]);
   while (true) {
      bool run_through = true;
      long i = n - 1;
      for (; i>=0; i--) {
         if (indices[i] != i + s - n){
            run_through = false;
            break;
         }
      }
      if (run_through) {
         return c;
      }
      indices[i]++;
      for (long j=i+1; j<(long)n; j++)
         indices[j] = indices[j-1] + 1;
      for(size_type k=0; k<n; k++)
         c[k].emplace_back(indices[k]);
   }
}

/// Return the indices of the elements which are not zero
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<double> v {2., 0., 3., 0., 1.};
/// auto nonzero_idx = Nonzero(v);
/// nonzero_idx
/// // (ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>::size_type>) { 0, 2, 4 }
/// ~~~
template <typename T>
RVec<typename RVec<T>::size_type> Nonzero(const RVec<T> &v)
{
   using size_type = typename RVec<T>::size_type;
   RVec<size_type> r;
   const auto size = v.size();
   r.reserve(size);
   for(size_type i=0; i<size; i++) {
      if(v[i] != 0) {
         r.emplace_back(i);
      }
   }
   return r;
}

/// Return the intersection of elements of two RVecs.
///
/// Each element of v1 is looked up in v2 and added to the returned vector if
/// found. Following, the order of v1 is preserved. If v2 is already sorted, the
/// optional argument v2_is_sorted can be used to toggle of the internal sorting
/// step, therewith optimising runtime.
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<double> v1 {1., 2., 3.};
/// RVec<double> v2 {-4., -5., 2., 1.};
/// auto v1_intersect_v2 = Intersect(v1, v2);
/// v1_intersect_v2
/// // (ROOT::VecOps::RVec<double>) { 1.0000000, 2.0000000 }
/// ~~~
template <typename T>
RVec<T> Intersect(const RVec<T>& v1, const RVec<T>& v2, bool v2_is_sorted = false)
{
   RVec<T> v2_sorted;
   if (!v2_is_sorted) v2_sorted = Sort(v2);
   const auto v2_begin = v2_is_sorted ? v2.begin() : v2_sorted.begin();
   const auto v2_end = v2_is_sorted ? v2.end() : v2_sorted.end();
   RVec<T> r;
   const auto size = v1.size();
   r.reserve(size);
   using size_type = typename RVec<T>::size_type;
   for(size_type i=0; i<size; i++) {
      if (std::binary_search(v2_begin, v2_end, v1[i])) {
         r.emplace_back(v1[i]);
      }
   }
   return r;
}

/// Return the elements of v1 if the condition c is true and v2 if the
/// condition c is false.
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<double> v1 {1., 2., 3.};
/// RVec<double> v2 {-1., -2., -3.};
/// auto c = v1 > 1;
/// c
/// // (ROOT::VecOps::RVec<int> &) { 0, 1, 1 }
/// auto if_c_v1_else_v2 = Where(c, v1, v2);
/// if_c_v1_else_v2
/// // (ROOT::VecOps::RVec<double> &) { -1.0000000, 2.0000000, 3.0000000 }
/// ~~~
template <typename T>
RVec<T> Where(const RVec<int>& c, const RVec<T>& v1, const RVec<T>& v2)
{
   using size_type = typename RVec<T>::size_type;
   const size_type size = c.size();
   RVec<T> r;
   r.reserve(size);
   for (size_type i=0; i<size; i++) {
      r.emplace_back(c[i] != 0 ? v1[i] : v2[i]);
   }
   return r;
}

/// Return the elements of v1 if the condition c is true and sets the value v2
/// if the condition c is false.
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<double> v1 {1., 2., 3.};
/// double v2 = 4.;
/// auto c = v1 > 1;
/// c
/// // (ROOT::VecOps::RVec<int> &) { 0, 1, 1 }
/// auto if_c_v1_else_v2 = Where(c, v1, v2);
/// if_c_v1_else_v2
/// // (ROOT::VecOps::RVec<double>) { 4.0000000, 2.0000000, 3.0000000 }
/// ~~~
template <typename T>
RVec<T> Where(const RVec<int>& c, const RVec<T>& v1, T v2)
{
   using size_type = typename RVec<T>::size_type;
   const size_type size = c.size();
   RVec<T> r;
   r.reserve(size);
   for (size_type i=0; i<size; i++) {
      r.emplace_back(c[i] != 0 ? v1[i] : v2);
   }
   return r;
}

/// Return the elements of v2 if the condition c is false and sets the value v1
/// if the condition c is true.
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// double v1 = 4.;
/// RVec<double> v2 {1., 2., 3.};
/// auto c = v2 > 1;
/// c
/// // (ROOT::VecOps::RVec<int> &) { 0, 1, 1 }
/// auto if_c_v1_else_v2 = Where(c, v1, v2);
/// if_c_v1_else_v2
/// // (ROOT::VecOps::RVec<double>) { 1.0000000, 4.0000000, 4.0000000 }
/// ~~~
template <typename T>
RVec<T> Where(const RVec<int>& c, T v1, const RVec<T>& v2)
{
   using size_type = typename RVec<T>::size_type;
   const size_type size = c.size();
   RVec<T> r;
   r.reserve(size);
   for (size_type i=0; i<size; i++) {
      r.emplace_back(c[i] != 0 ? v1 : v2[i]);
   }
   return r;
}

/// Return a vector with the value v2 if the condition c is false and sets the
/// value v1 if the condition c is true.
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// double v1 = 4.;
/// double v2 = 2.;
/// RVec<int> c {0, 1, 1};
/// auto if_c_v1_else_v2 = Where(c, v1, v2);
/// if_c_v1_else_v2
/// // (ROOT::VecOps::RVec<double>) { 2.0000000, 4.0000000, 4.0000000 }
/// ~~~
template <typename T>
RVec<T> Where(const RVec<int>& c, T v1, T v2)
{
   using size_type = typename RVec<T>::size_type;
   const size_type size = c.size();
   RVec<T> r;
   r.reserve(size);
   for (size_type i=0; i<size; i++) {
      r.emplace_back(c[i] != 0 ? v1 : v2);
   }
   return r;
}

/// Return the concatenation of two RVecs.
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<float> rvf {0.f, 1.f, 2.f};
/// RVec<int> rvi {7, 8, 9};
/// Concatenate(rvf, rvi);
/// // (ROOT::VecOps::RVec<float>) { 2.0000000, 4.0000000, 4.0000000 }
/// ~~~
template <typename T0, typename T1, typename Common_t = typename std::common_type<T0, T1>::type>
RVec<Common_t> Concatenate(const RVec<T0> &v0, const RVec<T1> &v1)
{
   RVec<Common_t> res;
   res.reserve(v0.size() + v1.size());
   auto &resAsVect = res.AsVector();
   auto &v0AsVect = v0.AsVector();
   auto &v1AsVect = v1.AsVector();
   resAsVect.insert(resAsVect.begin(), v0AsVect.begin(), v0AsVect.end());
   resAsVect.insert(resAsVect.end(), v1AsVect.begin(), v1AsVect.end());
   return res;
}

/// Return the angle difference \f$\Delta \phi\f$ of two scalars.
///
/// The function computes the closest angle from v1 to v2 with sign and is
/// therefore in the range \f$[-\pi, \pi]\f$.
/// The computation is done per default in radians \f$c = \pi\f$ but can be switched
/// to degrees \f$c = 180\f$.
template <typename T>
T DeltaPhi(T v1, T v2, const T c = M_PI)
{
   static_assert(std::is_floating_point<T>::value,
                 "DeltaPhi must be called with floating point values.");
   auto r = std::fmod(v2 - v1, 2.0 * c);
   if (r < -c) {
      r += 2.0 * c;
   }
   else if (r > c) {
      r -= 2.0 * c;
   }
   return r;
}

/// Return the angle difference \f$\Delta \phi\f$ in radians of two vectors.
///
/// The function computes the closest angle from v1 to v2 with sign and is
/// therefore in the range \f$[-\pi, \pi]\f$.
/// The computation is done per default in radians \f$c = \pi\f$ but can be switched
/// to degrees \f$c = 180\f$.
template <typename T>
RVec<T> DeltaPhi(const RVec<T>& v1, const RVec<T>& v2, const T c = M_PI)
{
   using size_type = typename RVec<T>::size_type;
   const size_type size = v1.size();
   auto r = RVec<T>(size);
   for (size_type i = 0; i < size; i++) {
      r[i] = DeltaPhi(v1[i], v2[i], c);
   }
   return r;
}

/// Return the angle difference \f$\Delta \phi\f$ in radians of a vector and a scalar.
///
/// The function computes the closest angle from v1 to v2 with sign and is
/// therefore in the range \f$[-\pi, \pi]\f$.
/// The computation is done per default in radians \f$c = \pi\f$ but can be switched
/// to degrees \f$c = 180\f$.
template <typename T>
RVec<T> DeltaPhi(const RVec<T>& v1, T v2, const T c = M_PI)
{
   using size_type = typename RVec<T>::size_type;
   const size_type size = v1.size();
   auto r = RVec<T>(size);
   for (size_type i = 0; i < size; i++) {
      r[i] = DeltaPhi(v1[i], v2, c);
   }
   return r;
}

/// Return the angle difference \f$\Delta \phi\f$ in radians of a scalar and a vector.
///
/// The function computes the closest angle from v1 to v2 with sign and is
/// therefore in the range \f$[-\pi, \pi]\f$.
/// The computation is done per default in radians \f$c = \pi\f$ but can be switched
/// to degrees \f$c = 180\f$.
template <typename T>
RVec<T> DeltaPhi(T v1, const RVec<T>& v2, const T c = M_PI)
{
   using size_type = typename RVec<T>::size_type;
   const size_type size = v2.size();
   auto r = RVec<T>(size);
   for (size_type i = 0; i < size; i++) {
      r[i] = DeltaPhi(v1, v2[i], c);
   }
   return r;
}

/// Return the square of the distance on the \f$\eta\f$-\f$\phi\f$ plane (\f$\Delta R\f$) from
/// the collections eta1, eta2, phi1 and phi2.
///
/// The function computes \f$\Delta R^2 = (\eta_1 - \eta_2)^2 + (\phi_1 - \phi_2)^2\f$
/// of the given collections eta1, eta2, phi1 and phi2. The angle \f$\phi\f$ can
/// be set to radian or degrees using the optional argument c, see the documentation
/// of the DeltaPhi helper.
template <typename T>
RVec<T> DeltaR2(const RVec<T>& eta1, const RVec<T>& eta2, const RVec<T>& phi1, const RVec<T>& phi2, const T c = M_PI)
{
   const auto dphi = DeltaPhi(phi1, phi2, c);
   return (eta1 - eta2) * (eta1 - eta2) + dphi * dphi;
}

/// Return the distance on the \f$\eta\f$-\f$\phi\f$ plane (\f$\Delta R\f$) from
/// the collections eta1, eta2, phi1 and phi2.
///
/// The function computes \f$\Delta R = \sqrt{(\eta_1 - \eta_2)^2 + (\phi_1 - \phi_2)^2}\f$
/// of the given collections eta1, eta2, phi1 and phi2. The angle \f$\phi\f$ can
/// be set to radian or degrees using the optional argument c, see the documentation
/// of the DeltaPhi helper.
template <typename T>
RVec<T> DeltaR(const RVec<T>& eta1, const RVec<T>& eta2, const RVec<T>& phi1, const RVec<T>& phi2, const T c = M_PI)
{
   return sqrt(DeltaR2(eta1, eta2, phi1, phi2, c));
}

/// Return the distance on the \f$\eta\f$-\f$\phi\f$ plane (\f$\Delta R\f$) from
/// the scalars eta1, eta2, phi1 and phi2.
///
/// The function computes \f$\Delta R = \sqrt{(\eta_1 - \eta_2)^2 + (\phi_1 - \phi_2)^2}\f$
/// of the given scalars eta1, eta2, phi1 and phi2. The angle \f$\phi\f$ can
/// be set to radian or degrees using the optional argument c, see the documentation
/// of the DeltaPhi helper.
template <typename T>
T DeltaR(T eta1, T eta2, T phi1, T phi2, const T c = M_PI)
{
   const auto dphi = DeltaPhi(phi1, phi2, c);
   return std::sqrt((eta1 - eta2) * (eta1 - eta2) + dphi * dphi);
}

/// Return the invariant mass of two particles given the collections of the quantities
/// transverse momentum (pt), rapidity (eta), azimuth (phi) and mass.
///
/// The function computes the invariant mass of two particles with the four-vectors
/// (pt1, eta2, phi1, mass1) and (pt2, eta2, phi2, mass2).
template <typename T>
RVec<T> InvariantMasses(
        const RVec<T>& pt1, const RVec<T>& eta1, const RVec<T>& phi1, const RVec<T>& mass1,
        const RVec<T>& pt2, const RVec<T>& eta2, const RVec<T>& phi2, const RVec<T>& mass2)
{
   std::size_t size = pt1.size();

   R__ASSERT(eta1.size() == size && phi1.size() == size && mass1.size() == size);
   R__ASSERT(pt2.size() == size && phi2.size() == size && mass2.size() == size);

   RVec<T> inv_masses(size);

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

/// Return the invariant mass of multiple particles given the collections of the
/// quantities transverse momentum (pt), rapidity (eta), azimuth (phi) and mass.
///
/// The function computes the invariant mass of multiple particles with the
/// four-vectors (pt, eta, phi, mass).
template <typename T>
T InvariantMass(const RVec<T>& pt, const RVec<T>& eta, const RVec<T>& phi, const RVec<T>& mass)
{
   const std::size_t size = pt.size();

   R__ASSERT(eta.size() == size && phi.size() == size && mass.size() == size);

   T x_sum = 0.;
   T y_sum = 0.;
   T z_sum = 0.;
   T e_sum = 0.;

   for (std::size_t i = 0u; i < size; ++ i) {
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

////////////////////////////////////////////////////////////////////////////
/// \brief Build an RVec of objects starting from RVecs of input to their constructors.
/// \tparam T Type of the objects contained in the created RVec.
/// \tparam Args_t Pack of types templating the input RVecs.
/// \param[in] args The RVecs containing the values used to initialise the output objects.
/// \return The RVec of objects initialised with the input parameters.
///
/// Example code, at the ROOT prompt:
/// ~~~{.cpp}
/// using namespace ROOT::VecOps;
/// RVec<float> pts = {15.5, 34.32, 12.95};
/// RVec<float> etas = {0.3, 2.2, 1.32};
/// RVec<float> phis = {0.1, 3.02, 2.2};
/// RVec<float> masses = {105.65, 105.65, 105.65};
/// auto fourVecs = Construct<ROOT::Math::PtEtaPhiMVector>(pts, etas, phis, masses);
/// cout << fourVecs << endl;
/// // { (15.5,0.3,0.1,105.65), (34.32,2.2,3.02,105.65), (12.95,1.32,2.2,105.65) }
/// ~~~
template <typename T, typename... Args_t>
RVec<T> Construct(const RVec<Args_t> &... args)
{
   const auto size = ::ROOT::GetVectorsSize("Construct", args...);
   RVec<T> ret;
   ret.reserve(size);
   for (auto i = 0UL; i < size; ++i) {
      ret.emplace_back(args[i]...);
   }
   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Print a RVec at the prompt:
template <class T>
std::ostream &operator<<(std::ostream &os, const RVec<T> &v)
{
   // In order to print properly, convert to 64 bit int if this is a char
   constexpr bool mustConvert = std::is_same<char, T>::value || std::is_same<signed char, T>::value ||
                                std::is_same<unsigned char, T>::value || std::is_same<wchar_t, T>::value ||
                                std::is_same<char16_t, T>::value || std::is_same<char32_t, T>::value;
   using Print_t = typename std::conditional<mustConvert, long long int, T>::type;
   os << "{ ";
   auto size = v.size();
   if (size) {
      for (std::size_t i = 0; i < size - 1; ++i) {
         os << (Print_t)v[i] << ", ";
      }
      os << (Print_t)v[size - 1];
   }
   os << " }";
   return os;
}

#if (_VECOPS_USE_EXTERN_TEMPLATES)

#define RVEC_EXTERN_UNARY_OPERATOR(T, OP) \
   extern template RVec<T> operator OP<T>(const RVec<T> &);

#define RVEC_EXTERN_BINARY_OPERATOR(T, OP)                                     \
   extern template auto operator OP<T, T>(const T &x, const RVec<T> &v)        \
      -> RVec<decltype(x OP v[0])>;                                            \
   extern template auto operator OP<T, T>(const RVec<T> &v, const T &y)        \
      -> RVec<decltype(v[0] OP y)>;                                            \
   extern template auto operator OP<T, T>(const RVec<T> &v0, const RVec<T> &v1)\
      -> RVec<decltype(v0[0] OP v1[0])>;

#define RVEC_EXTERN_ASSIGN_OPERATOR(T, OP)                           \
   extern template RVec<T> &operator OP<T, T>(RVec<T> &, const T &); \
   extern template RVec<T> &operator OP<T, T>(RVec<T> &, const RVec<T> &);

#define RVEC_EXTERN_LOGICAL_OPERATOR(T, OP)                                 \
   extern template RVec<int> operator OP<T, T>(const RVec<T> &, const T &); \
   extern template RVec<int> operator OP<T, T>(const T &, const RVec<T> &); \
   extern template RVec<int> operator OP<T, T>(const RVec<T> &, const RVec<T> &);

#define RVEC_EXTERN_FLOAT_TEMPLATE(T)   \
   extern template class RVec<T>;       \
   RVEC_EXTERN_UNARY_OPERATOR(T, +)     \
   RVEC_EXTERN_UNARY_OPERATOR(T, -)     \
   RVEC_EXTERN_UNARY_OPERATOR(T, !)     \
   RVEC_EXTERN_BINARY_OPERATOR(T, +)    \
   RVEC_EXTERN_BINARY_OPERATOR(T, -)    \
   RVEC_EXTERN_BINARY_OPERATOR(T, *)    \
   RVEC_EXTERN_BINARY_OPERATOR(T, /)    \
   RVEC_EXTERN_ASSIGN_OPERATOR(T, +=)   \
   RVEC_EXTERN_ASSIGN_OPERATOR(T, -=)   \
   RVEC_EXTERN_ASSIGN_OPERATOR(T, *=)   \
   RVEC_EXTERN_ASSIGN_OPERATOR(T, /=)   \
   RVEC_EXTERN_LOGICAL_OPERATOR(T, <)   \
   RVEC_EXTERN_LOGICAL_OPERATOR(T, >)   \
   RVEC_EXTERN_LOGICAL_OPERATOR(T, ==)  \
   RVEC_EXTERN_LOGICAL_OPERATOR(T, !=)  \
   RVEC_EXTERN_LOGICAL_OPERATOR(T, <=)  \
   RVEC_EXTERN_LOGICAL_OPERATOR(T, >=)  \
   RVEC_EXTERN_LOGICAL_OPERATOR(T, &&)  \
   RVEC_EXTERN_LOGICAL_OPERATOR(T, ||)

#define RVEC_EXTERN_INTEGER_TEMPLATE(T) \
   extern template class RVec<T>;       \
   RVEC_EXTERN_UNARY_OPERATOR(T, +)     \
   RVEC_EXTERN_UNARY_OPERATOR(T, -)     \
   RVEC_EXTERN_UNARY_OPERATOR(T, ~)     \
   RVEC_EXTERN_UNARY_OPERATOR(T, !)     \
   RVEC_EXTERN_BINARY_OPERATOR(T, +)    \
   RVEC_EXTERN_BINARY_OPERATOR(T, -)    \
   RVEC_EXTERN_BINARY_OPERATOR(T, *)    \
   RVEC_EXTERN_BINARY_OPERATOR(T, /)    \
   RVEC_EXTERN_BINARY_OPERATOR(T, %)    \
   RVEC_EXTERN_BINARY_OPERATOR(T, &)    \
   RVEC_EXTERN_BINARY_OPERATOR(T, |)    \
   RVEC_EXTERN_BINARY_OPERATOR(T, ^)    \
   RVEC_EXTERN_ASSIGN_OPERATOR(T, +=)   \
   RVEC_EXTERN_ASSIGN_OPERATOR(T, -=)   \
   RVEC_EXTERN_ASSIGN_OPERATOR(T, *=)   \
   RVEC_EXTERN_ASSIGN_OPERATOR(T, /=)   \
   RVEC_EXTERN_ASSIGN_OPERATOR(T, %=)   \
   RVEC_EXTERN_ASSIGN_OPERATOR(T, &=)   \
   RVEC_EXTERN_ASSIGN_OPERATOR(T, |=)   \
   RVEC_EXTERN_ASSIGN_OPERATOR(T, ^=)   \
   RVEC_EXTERN_ASSIGN_OPERATOR(T, >>=)  \
   RVEC_EXTERN_ASSIGN_OPERATOR(T, <<=)  \
   RVEC_EXTERN_LOGICAL_OPERATOR(T, <)   \
   RVEC_EXTERN_LOGICAL_OPERATOR(T, >)   \
   RVEC_EXTERN_LOGICAL_OPERATOR(T, ==)  \
   RVEC_EXTERN_LOGICAL_OPERATOR(T, !=)  \
   RVEC_EXTERN_LOGICAL_OPERATOR(T, <=)  \
   RVEC_EXTERN_LOGICAL_OPERATOR(T, >=)  \
   RVEC_EXTERN_LOGICAL_OPERATOR(T, &&)  \
   RVEC_EXTERN_LOGICAL_OPERATOR(T, ||)

RVEC_EXTERN_INTEGER_TEMPLATE(char)
RVEC_EXTERN_INTEGER_TEMPLATE(short)
RVEC_EXTERN_INTEGER_TEMPLATE(int)
RVEC_EXTERN_INTEGER_TEMPLATE(long)
//RVEC_EXTERN_INTEGER_TEMPLATE(long long)

RVEC_EXTERN_INTEGER_TEMPLATE(unsigned char)
RVEC_EXTERN_INTEGER_TEMPLATE(unsigned short)
RVEC_EXTERN_INTEGER_TEMPLATE(unsigned int)
RVEC_EXTERN_INTEGER_TEMPLATE(unsigned long)
//RVEC_EXTERN_INTEGER_TEMPLATE(unsigned long long)

RVEC_EXTERN_FLOAT_TEMPLATE(float)
RVEC_EXTERN_FLOAT_TEMPLATE(double)

#undef RVEC_EXTERN_UNARY_OPERATOR
#undef RVEC_EXTERN_BINARY_OPERATOR
#undef RVEC_EXTERN_ASSIGN_OPERATOR
#undef RVEC_EXTERN_LOGICAL_OPERATOR
#undef RVEC_EXTERN_INTEGER_TEMPLATE
#undef RVEC_EXTERN_FLOAT_TEMPLATE

#define RVEC_EXTERN_UNARY_FUNCTION(T, NAME, FUNC) \
   extern template RVec<PromoteType<T>> NAME(const RVec<T> &);

#define RVEC_EXTERN_STD_UNARY_FUNCTION(T, F) RVEC_EXTERN_UNARY_FUNCTION(T, F, std::F)

#define RVEC_EXTERN_BINARY_FUNCTION(T0, T1, NAME, FUNC)                            \
   extern template RVec<PromoteTypes<T0, T1>> NAME(const RVec<T0> &, const T1 &); \
   extern template RVec<PromoteTypes<T0, T1>> NAME(const T0 &, const RVec<T1> &); \
   extern template RVec<PromoteTypes<T0, T1>> NAME(const RVec<T0> &, const RVec<T1> &);

#define RVEC_EXTERN_STD_BINARY_FUNCTION(T, F) RVEC_EXTERN_BINARY_FUNCTION(T, T, F, std::F)

#define RVEC_EXTERN_STD_FUNCTIONS(T)             \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, abs)        \
   RVEC_EXTERN_STD_BINARY_FUNCTION(T, fdim)      \
   RVEC_EXTERN_STD_BINARY_FUNCTION(T, fmod)      \
   RVEC_EXTERN_STD_BINARY_FUNCTION(T, remainder) \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, exp)        \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, exp2)       \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, expm1)      \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, log)        \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, log10)      \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, log2)       \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, log1p)      \
   RVEC_EXTERN_STD_BINARY_FUNCTION(T, pow)       \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, sqrt)       \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, cbrt)       \
   RVEC_EXTERN_STD_BINARY_FUNCTION(T, hypot)     \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, sin)        \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, cos)        \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, tan)        \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, asin)       \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, acos)       \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, atan)       \
   RVEC_EXTERN_STD_BINARY_FUNCTION(T, atan2)     \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, sinh)       \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, cosh)       \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, tanh)       \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, asinh)      \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, acosh)      \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, atanh)      \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, floor)      \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, ceil)       \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, trunc)      \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, round)      \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, erf)        \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, erfc)       \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, lgamma)     \
   RVEC_EXTERN_STD_UNARY_FUNCTION(T, tgamma)     \

RVEC_EXTERN_STD_FUNCTIONS(float)
RVEC_EXTERN_STD_FUNCTIONS(double)
#undef RVEC_EXTERN_STD_UNARY_FUNCTION
#undef RVEC_EXTERN_STD_BINARY_FUNCTION
#undef RVEC_EXTERN_STD_UNARY_FUNCTIONS

#ifdef R__HAS_VDT

#define RVEC_EXTERN_VDT_UNARY_FUNCTION(T, F) RVEC_EXTERN_UNARY_FUNCTION(T, F, vdt::F)

RVEC_EXTERN_VDT_UNARY_FUNCTION(float, fast_expf)
RVEC_EXTERN_VDT_UNARY_FUNCTION(float, fast_logf)
RVEC_EXTERN_VDT_UNARY_FUNCTION(float, fast_sinf)
RVEC_EXTERN_VDT_UNARY_FUNCTION(float, fast_cosf)
RVEC_EXTERN_VDT_UNARY_FUNCTION(float, fast_tanf)
RVEC_EXTERN_VDT_UNARY_FUNCTION(float, fast_asinf)
RVEC_EXTERN_VDT_UNARY_FUNCTION(float, fast_acosf)
RVEC_EXTERN_VDT_UNARY_FUNCTION(float, fast_atanf)

RVEC_EXTERN_VDT_UNARY_FUNCTION(double, fast_exp)
RVEC_EXTERN_VDT_UNARY_FUNCTION(double, fast_log)
RVEC_EXTERN_VDT_UNARY_FUNCTION(double, fast_sin)
RVEC_EXTERN_VDT_UNARY_FUNCTION(double, fast_cos)
RVEC_EXTERN_VDT_UNARY_FUNCTION(double, fast_tan)
RVEC_EXTERN_VDT_UNARY_FUNCTION(double, fast_asin)
RVEC_EXTERN_VDT_UNARY_FUNCTION(double, fast_acos)
RVEC_EXTERN_VDT_UNARY_FUNCTION(double, fast_atan)

#endif // R__HAS_VDT

#endif // _VECOPS_USE_EXTERN_TEMPLATES

} // namespace ROOT
