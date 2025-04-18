cmake_minimum_required(VERSION 3.15)
project(ComputationalGraph)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3")

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/Release/generators")
find_package(math REQUIRED CONFIG)
target_link_libraries(ComputationalGraph PUBLIC math::math)

add_library(ComputationalGraph src/Sigmoid.cpp
        src/ReLU.cpp
        src/Softmax.cpp
        src/Tanh.cpp
        src/Function.h
        src/Sigmoid.h
        src/ReLU.h
        src/Softmax.h
        src/Tanh.h
        src/FunctionType.h
        src/ComputationalNode.cpp
        src/ComputationalGraph.cpp
        src/ComputationalGraphWithTensor.cpp
        src/ComputationalGraph.h
        src/ComputationalGraphWithTensor.h
        src/ComputationalNode.h)

target_link_libraries(ComputationalGraph math::math)



