include(FindGBench)
FindGBench()

add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE benchmark::benchmark)
