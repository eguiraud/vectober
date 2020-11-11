// Copyright (C) 2020 Enrico Guiraud (see LICENSE file in top directory)
// Author: Enrico Guiraud

#include <benchmark/benchmark.h>
#include <SmallVector.h>

#include "vec_operators.hpp"
#include "benchmarks.hpp"

BENCHMARK_TEMPLATE(Assign, llvm::SmallVector<float, 8>)->Apply(GetArguments);
BENCHMARK_TEMPLATE(Add, llvm::SmallVector<float, 8>)->Apply(GetArguments);
BENCHMARK_TEMPLATE(SquareAddSqrt, llvm::SmallVector<float, 8>)->Apply(GetArguments);
BENCHMARK_TEMPLATE(DeltaR, llvm::SmallVector<float, 8>)->Apply(GetArguments);
BENCHMARK_TEMPLATE(InvariantMass, llvm::SmallVector<float, 8>)->Apply(GetArguments);
BENCHMARK_TEMPLATE(InvariantMasses, llvm::SmallVector<float, 8>)->Apply(GetArguments);
BENCHMARK_TEMPLATE(MaskingSimple, llvm::SmallVector<float, 8>)->Apply(GetArguments);
BENCHMARK_TEMPLATE(MaskingComplex, llvm::SmallVector<float, 8>)->Apply(GetArguments);

// trigger instantiation of SmallVector<NonPOD> to make sure it compiles
void instantiate_nonpod_smallvector() {
   struct S {
      int x;
      S() { x = 42; }
      S(const S&) { x = 3; }
      ~S() { x = 0; }
   };

   static_assert(!std::is_trivially_copy_constructible<S>::value, "");

   llvm::SmallVector<S, 8> v;
   v.push_back(S()); // trigger instantiation of `grow` for non-POD types
}

BENCHMARK_MAIN();
