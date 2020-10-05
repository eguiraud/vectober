function(FindGBench)
   find_package(benchmark QUIET)

   if (NOT benchmark_FOUND)
      message(STATUS "Could not find a local installation of google benchmark.")
      message(STATUS "It will be fetched from GitHub and built together with the project.")
      include(FetchContent)
      FetchContent_Declare(
        googlebench
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.5.2
      )
      FetchContent_Populate(googlebench)
      set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "Download gtest if needed")
      set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable building of gbench tests")
      set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable installation of gbench (it's a private dependency)")
      add_subdirectory(${googlebench_SOURCE_DIR} ${googlebench_BINARY_DIR})
      add_library(benchmark::benchmark ALIAS benchmark)
   endif()
endfunction()
