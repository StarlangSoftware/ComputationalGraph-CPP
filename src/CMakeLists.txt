cmake_minimum_required(VERSION 3.22)
project(ComputationalGraph)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3")

find_package(math REQUIRED)
find_package(classification REQUIRED)

add_library(ComputationalGraph Function/Sigmoid.cpp
        Function/ReLU.cpp
        Function/Softmax.cpp
        Function/Tanh.cpp
        Function/Function.h
        Function/Sigmoid.h
        Function/ReLU.h
        Function/Softmax.h
        Function/Tanh.h
        Function/FunctionType.h
        ComputationalGraph.cpp
        ComputationalGraph.h
        Function/DELU.cpp
        Function/DELU.h
        Function/Dropout.cpp
        Function/Dropout.h
        Function/ELU.cpp
        Function/ELU.h
        Function/Negation.cpp
        Function/Negation.h
        Initialization/Initialization.h
        Initialization/HeUniformInitialization.cpp
        Initialization/HeUniformInitialization.h
        Initialization/RandomInitialization.cpp
        Initialization/RandomInitialization.h
        Initialization/UniformXavierInitialization.cpp
        Initialization/UniformXavierInitialization.h
        Node/ComputationalNode.cpp
        Node/ComputationalNode.h
        Node/ConcatenatedNode.cpp
        Node/ConcatenatedNode.h
        Node/MultiplicationNode.cpp
        Node/MultiplicationNode.h
        Optimizer/Optimizer.cpp
        Optimizer/Optimizer.h
        Optimizer/StochasticGradientDescent.cpp
        Optimizer/StochasticGradientDescent.h
        Optimizer/SGDMomentum.cpp
        Optimizer/SGDMomentum.h
        Optimizer/Adam.cpp
        Optimizer/Adam.h
        Optimizer/AdamW.cpp
        Optimizer/AdamW.h
        NeuralNetworkParameter.cpp
        NeuralNetworkParameter.h
)

target_link_libraries(ComputationalGraph math::math classification::classification)
