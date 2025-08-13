cmake_minimum_required(VERSION 3.16)
project(TensorRT_CXX_Inference_Engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Clang as the host compiler for nvcc
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/clang-14)

# Enable CUDA language
enable_language(CUDA)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# Add cmake modules path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Find required packages
find_package(CUDA REQUIRED)

# Try to find TensorRT - it might not be in standard locations
# You may need to set TensorRT_DIR manually
if(NOT DEFINED TensorRT_DIR)
    # Common TensorRT installation paths
    set(TensorRT_DIR "/path/to/TensorRT-10.9.0.34" CACHE PATH "TensorRT installation directory")
endif()

find_package(TensorRT)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TensorRT_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})

add_library(TensorRT_CXX_Inference_Engine STATIC
    TRT_inference_engine.cpp
    TRT_inference_engine.hpp
)
target_include_directories(TensorRT_CXX_Inference_Engine PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${TensorRT_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(TensorRT_CXX_Inference_Engine
    ${TensorRT_LIBRARIES}
    ${CUDA_LIBRARIES}
)
