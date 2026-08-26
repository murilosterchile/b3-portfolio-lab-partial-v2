cmake_minimum_required(VERSION 3.20)
project(qkp_exact LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(qkp_exact src/solver.cpp)
target_include_directories(qkp_exact PUBLIC include)
target_compile_options(qkp_exact PRIVATE -Wall -Wextra -Wpedantic)

add_executable(qkp_cli src/main.cpp)
target_link_libraries(qkp_cli PRIVATE qkp_exact)

add_executable(qkp_tests tests/test_solver.cpp)
target_link_libraries(qkp_tests PRIVATE qkp_exact)

enable_testing()
add_test(NAME qkp_tests COMMAND qkp_tests)
