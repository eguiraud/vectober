add_library(ROOTVecOps
            SHARED
            RAdoptAllocator.hxx
            RIntegerSequence.hxx
            RVec.hxx
            RVec.cxx
            TypeTraits.hxx)

add_library(LLVMSmallVector
            SHARED
            SmallVector.h
            SmallVector.cpp
            AlignOf.h
            Compiler.h
            iterator_range.h)
