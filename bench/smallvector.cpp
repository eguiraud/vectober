// Copyright (C) 2020 Enrico Guiraud (see LICENSE file in top directory)
// Author: Enrico Guiraud

#include <benchmark/benchmark.h>
#include <SmallVector.h>

#include "benchmarks.hpp"
#include "vec_operators.hpp"

BENCHMARK_TEMPLATE(Assign, llvm::SmallVector<float, 8>)->Apply(GetArguments);
BENCHMARK_TEMPLATE(Add, llvm::SmallVector<float, 8>)->Apply(GetArguments);
// BENCHMARK_TEMPLATE(SquareAddSqrt, llvm::SmallVector<float, 8>)->Apply(GetArguments);
// BENCHMARK_TEMPLATE(DeltaR, llvm::SmallVector<float, 8>)->Apply(GetArguments);
// BENCHMARK_TEMPLATE(InvariantMass, llvm::SmallVector<float, 8>)->Apply(GetArguments);
// BENCHMARK_TEMPLATE(InvariantMasses, llvm::SmallVector<float, 8>)->Apply(GetArguments);
// BENCHMARK_TEMPLATE(MaskingSimple, llvm::SmallVector<float, 8>)->Apply(GetArguments);
// BENCHMARK_TEMPLATE(MaskingComplex, llvm::SmallVector<float, 8>)->Apply(GetArguments);

BENCHMARK_MAIN();
