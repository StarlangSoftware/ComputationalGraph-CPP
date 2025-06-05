cmake_minimum_required(VERSION 3.12)
project(ComputationalGraph)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3")

find_package(math REQUIRED)
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
        src/ComputationalGraph.cpp
        src/ComputationalGraph.h
)

target_link_libraries(ComputationalGraph math::math)
add_executable(Test src/Sigmoid.cpp
        src/ReLU.cpp
        src/Softmax.cpp
        src/Tanh.cpp
        src/Function.h
        src/Sigmoid.h
        src/ReLU.h
        src/Softmax.h
        src/Tanh.h
        src/FunctionType.h
        src/ComputationalGraph.cpp
        src/ComputationalGraph.h
)
target_link_libraries(Test math::math)



