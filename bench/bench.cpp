// Copyright (C) 2020 Enrico Guiraud (see LICENSE file in top directory)

#include <benchmark/benchmark.h>
#include <string>

static void BM_StringCreation(benchmark::State &state)
{
   for (auto _ : state)
      std::string empty_string;
}
BENCHMARK(BM_StringCreation);

static void BM_StringCopy(benchmark::State &state)
{
   std::string x = "hello";
   for (auto _ : state)
      std::string copy(x);
}
BENCHMARK(BM_StringCopy);

BENCHMARK_MAIN();
