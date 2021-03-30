cmake_minimum_required(VERSION 3.17)
project(2_5GaussSeidel)

add_executable(2_5GaussSeidel 2_5GaussSeidel.cpp)

#Acesso ao GSL-GNU
target_link_libraries(2_5GaussSeidel -lgsl -lgslcblas -lm)

#Acesso ao OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")
set(SOURCE_FILES 2_5GaussSeidel.cpp)