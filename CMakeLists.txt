cmake_minimum_required(VERSION 3.11)
project (vectober CXX)
set(CMAKE_CXX_STANDARD 11)
add_compile_options(-Wall -Wextra -Wpedantic)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
   message(STATUS "Build type not specified, defaulting to ${CMAKE_BUILD_TYPE}")
endif()

add_subdirectory(bench)
