cmake_minimum_required(VERSION 3.15)
project(ComputationalGraph)

# Conan setup
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
        src/Sigmoid.cpp
        src/ReLU.cpp
        src/Softmax.cpp
        )

# Header files
set(HEADERS
        include/ComputationalGraph/Function.hpp
        include/ComputationalGraph/Sigmoid.hpp
        include/ComputationalGraph/ReLU.hpp
        include/ComputationalGraph/Softmax.hpp
        include/ComputationalGraph/Tanh.hpp
        )

# Library creation
add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Linking Conan dependencies
target_link_libraries(${PROJECT_NAME} ${CONAN_LIBS})

# C++ standard
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
