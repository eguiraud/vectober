include(FindGBench)
FindGBench()

add_executable(rvec rvec.cpp)
target_include_directories(rvec PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(rvec PRIVATE benchmark::benchmark ROOTVecOps)
