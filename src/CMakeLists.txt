cmake_minimum_required(VERSION 3.15)
project(ComputationalGraph)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3")

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/Release/generators")
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
        ComputationalNode.cpp
        ComputationalGraph.cpp
        ComputationalGraphWithTensor.cpp
        ComputationalGraph.h
        ComputationalGraphWithTensor.h
        ComputationalNode.h)

target_link_libraries(ComputationalGraph math::math)

