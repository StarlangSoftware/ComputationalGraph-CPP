cmake_minimum_required(VERSION 3.12)
project(ComputationalGraph)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3")

find_package(math REQUIRED)
find_package(classification REQUIRED)

set(SOURCE_FILES
        src/Function/Sigmoid.cpp
        src/Function/ReLU.cpp
        src/Function/Softmax.cpp
        src/Function/Tanh.cpp
        src/Function/Function.h
        src/Function/Sigmoid.h
        src/Function/ReLU.h
        src/Function/Softmax.h
        src/Function/Tanh.h
        src/Function/FunctionType.h
        src/ComputationalGraph.cpp
        src/ComputationalGraph.h
        src/Function/DELU.cpp
        src/Function/DELU.h
        src/Function/Dropout.cpp
        src/Function/Dropout.h
        src/Function/ELU.cpp
        src/Function/ELU.h
        src/Function/Negation.cpp
        src/Function/Negation.h
        src/Function/Logarithm.cpp
        src/Function/Logarithm.h
        src/Function/Power.cpp
        src/Function/Power.h
        src/Function/SiLU.cpp
        src/Function/SiLU.h
        src/Function/TanhShrink.cpp
        src/Function/TanhShrink.h
        src/Function/CrossEntropyLoss.cpp
        src/Function/CrossEntropyLoss.h
        src/Function/MeanSquaredErrorLoss.cpp
        src/Function/MeanSquaredErrorLoss.h
        src/Initialization/Initialization.h
        src/Initialization/HeUniformInitialization.cpp
        src/Initialization/HeUniformInitialization.h
        src/Initialization/RandomInitialization.cpp
        src/Initialization/RandomInitialization.h
        src/Initialization/UniformXavierInitialization.cpp
        src/Initialization/UniformXavierInitialization.h
        src/Node/ComputationalNode.cpp
        src/Node/ComputationalNode.h
        src/Node/ConcatenatedNode.cpp
        src/Node/ConcatenatedNode.h
        src/Node/MultiplicationNode.cpp
        src/Node/MultiplicationNode.h
        src/Optimizer/Optimizer.cpp
        src/Optimizer/Optimizer.h
        src/Optimizer/StochasticGradientDescent.cpp
        src/Optimizer/StochasticGradientDescent.h
        src/Optimizer/SGDMomentum.cpp
        src/Optimizer/SGDMomentum.h
        src/Optimizer/Adam.cpp
        src/Optimizer/Adam.h
        src/Optimizer/AdamW.cpp
        src/Optimizer/AdamW.h
        src/NeuralNetworkParameter.cpp
        src/NeuralNetworkParameter.h
)

add_library(ComputationalGraph ${SOURCE_FILES})
target_link_libraries(ComputationalGraph math::math classification::classification)

add_executable(CGTest 
        ${SOURCE_FILES}
        test/LinearPerceptronSingleUnit.cpp
        test/LinearPerceptronSingleUnit.h
        test/ComputationalGraphTest.cpp
        test/NeuralNetwork.cpp
        test/NeuralNetwork.h
        test/LinearPerceptron.cpp
        test/LinearPerceptron.h
        test/MultiLayerPerceptron.cpp
        test/MultiLayerPerceptron.h
        test/DeepNetwork.cpp
        test/DeepNetwork.h
)

target_link_libraries(CGTest math::math classification::classification)