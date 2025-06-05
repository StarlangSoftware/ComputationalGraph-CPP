cmake_minimum_required(VERSION 3.22)
project(ComputationalGraph)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3")

find_package(math REQUIRED)

add_library(ComputationalGraph Sigmoid.cpp
        ReLU.cpp
        Softmax.cpp
        Tanh.cpp
        Function.h
        Sigmoid.h
        ReLU.h
        Softmax.h
        Tanh.h
        FunctionType.h
        ComputationalGraph.cpp
        ComputationalGraph.h
)

target_link_libraries(ComputationalGraph math::math)
