include(FindGBench)
FindGBench()

add_executable(bench bench.cpp)
target_include_directories(bench PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(bench PRIVATE benchmark::benchmark ROOTVecOps)
