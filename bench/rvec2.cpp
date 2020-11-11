// Copyright (C) 2020 Enrico Guiraud (see LICENSE file in top directory)
// Author: Enrico Guiraud, Stefan Wunsch

#include <benchmark/benchmark.h>
#include <RVec2.h>

#include "RVec2Operators.h"
#include "benchmarks.hpp"

using namespace ROOT;

BENCHMARK_TEMPLATE(Assign, RVec<float>)->Apply(GetArguments);
BENCHMARK_TEMPLATE(Add, RVec<float>)->Apply(GetArguments);
BENCHMARK_TEMPLATE(SquareAddSqrt, RVec<float>)->Apply(GetArguments);
BENCHMARK_TEMPLATE(DeltaR, RVec<float>)->Apply(GetArguments);
BENCHMARK_TEMPLATE(InvariantMass, RVec<float>)->Apply(GetArguments);
BENCHMARK_TEMPLATE(InvariantMasses, RVec<float>)->Apply(GetArguments);
BENCHMARK_TEMPLATE(MaskingSimple, RVec<float>)->Apply(GetArguments);
BENCHMARK_TEMPLATE(MaskingComplex, RVec<float>)->Apply(GetArguments);

BENCHMARK_MAIN();
