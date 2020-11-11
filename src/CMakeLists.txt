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
            SmallVector.cpp)

add_library(RVec2
            SHARED
            RVec2.h
            RVec2.cpp)
