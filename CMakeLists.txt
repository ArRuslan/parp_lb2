cmake_minimum_required(VERSION 3.29)
project(parp_lb2)

set(CMAKE_CXX_STANDARD 20)

add_executable(parp_lb2 main.cpp)
target_link_libraries(parp_lb2 PRIVATE gomp)
target_compile_options(parp_lb2 PRIVATE -mavx -msse -msse2 -msse4.1 -march=native) # i have no idea which flag made it compile and atp i dont want to try finding one
