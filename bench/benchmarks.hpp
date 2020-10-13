// Copyright (C) 2020 Enrico Guiraud (see LICENSE file in top directory)
// Authors: Enrico Guiraud, Stefan Wunsch

#include <benchmark/benchmark.h>

static void GetArguments(benchmark::internal::Benchmark *b)
{
   for (const auto i : {1, 2, 4, 8, 16, 32, 64})
      b->Arg(i);
}

template <class Vec_t>
static void Assign(benchmark::State &state)
{
   const auto size = state.range(0);
   Vec_t v1(size);
   Vec_t v2(size);
   for (auto _ : state) {
      v2 = v1;
      benchmark::DoNotOptimize(v2);
   }
}

template <class Vec_t>
static void Add(benchmark::State &state)
{
   const auto size = state.range(0);
   Vec_t v1(size);
   Vec_t v2(size);
   for (auto _ : state) {
      benchmark::DoNotOptimize(v1 + v2);
   }
}

template <class Vec_t>
static void SquareAddSqrt(benchmark::State &state)
{
   const auto size = state.range(0);
   Vec_t v1(size);
   Vec_t v2(size);
   for (auto _ : state) {
      benchmark::DoNotOptimize(sqrt(pow(v1, 2) + pow(v2, 2)));
   }
}

template <class Vec_t>
static void DeltaR(benchmark::State &state)
{
   const auto size = state.range(0);
   Vec_t eta1(size);
   Vec_t eta2(size);
   Vec_t phi1(size);
   Vec_t phi2(size);
   for (auto _ : state) {
      benchmark::DoNotOptimize(DeltaR(eta1, eta2, phi1, phi2));
   }
}

template <class Vec_t>
static void InvariantMass(benchmark::State &state)
{
   const auto size = state.range(0);
   Vec_t pt(size);
   Vec_t eta(size);
   Vec_t phi(size);
   Vec_t mass(size);
   for (auto _ : state) {
      benchmark::DoNotOptimize(InvariantMass(pt, eta, phi, mass));
   }
}

template <class Vec_t>
static void InvariantMasses(benchmark::State &state)
{
   const auto size = state.range(0);
   Vec_t pt1(size);
   Vec_t pt2(size);
   Vec_t eta1(size);
   Vec_t eta2(size);
   Vec_t phi1(size);
   Vec_t phi2(size);
   Vec_t mass1(size);
   Vec_t mass2(size);
   for (auto _ : state) {
      benchmark::DoNotOptimize(InvariantMasses(pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2));
   }
}

template <class Vec_t>
Vec_t GenerateSeq(const unsigned int size)
{
   Vec_t v(size);
   for (auto i = 0u; i < size; i++)
      v[i] = i;
   return v;
}

template <class Vec_t>
static void MaskingSimple(benchmark::State &state)
{
   const auto size = state.range(0);
   auto v = GenerateSeq<Vec_t>(size);
   for (auto _ : state) {
      benchmark::DoNotOptimize(v[v > 0.5 * size]);
   }
}

template <class Vec_t>
static void MaskingComplex(benchmark::State &state)
{
   const auto size = state.range(0);
   auto v1 = GenerateSeq<Vec_t>(size);
   auto v2 = GenerateSeq<Vec_t>(size);
   for (auto _ : state) {
      benchmark::DoNotOptimize(v2[v1 > 0.5 * size && v2 < 0.75 * size && v1 + v2 > 0]);
   }
}
