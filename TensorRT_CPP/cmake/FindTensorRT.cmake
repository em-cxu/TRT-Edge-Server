# FindTensorRT.cmake
# Locates the TensorRT library
#
# This module defines the following variables:
#   TensorRT_FOUND - True if TensorRT is found
#   TensorRT_INCLUDE_DIRS - TensorRT include directories
#   TensorRT_LIBRARIES - TensorRT libraries
#   TensorRT_VERSION - TensorRT version

# Common TensorRT installation paths
set(TensorRT_SEARCH_PATHS
    "${TensorRT_DIR}"
    "/usr/local/TensorRT"
    "/usr/lib/x86_64-linux-gnu"
    "/opt/TensorRT"
    "/usr/local/cuda/TensorRT"
    "$ENV{TENSORRT_ROOT}"
    "$ENV{TENSORRT_HOME}"
)

# Find TensorRT include directory
find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${TensorRT_SEARCH_PATHS}
    PATH_SUFFIXES include
    DOC "TensorRT include directory"
)

# Find TensorRT libraries
find_library(TensorRT_LIBRARY
    NAMES nvinfer
    PATHS ${TensorRT_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
    DOC "TensorRT library"
)

find_library(TensorRT_PLUGIN_LIBRARY
    NAMES nvinfer_plugin
    PATHS ${TensorRT_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
    DOC "TensorRT plugin library"
)

# Find TensorRT version
if(TensorRT_INCLUDE_DIR)
    file(READ "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TENSORRT_VERSION_H)
    string(REGEX MATCH "#define NV_TENSORRT_MAJOR ([0-9]+)" TENSORRT_MAJOR_MATCH ${TENSORRT_VERSION_H})
    string(REGEX MATCH "#define NV_TENSORRT_MINOR ([0-9]+)" TENSORRT_MINOR_MATCH ${TENSORRT_VERSION_H})
    string(REGEX MATCH "#define NV_TENSORRT_PATCH ([0-9]+)" TENSORRT_PATCH_MATCH ${TENSORRT_VERSION_H})
    
    if(TENSORRT_MAJOR_MATCH AND TENSORRT_MINOR_MATCH AND TENSORRT_PATCH_MATCH)
        set(TensorRT_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
    endif()
endif()

# Set TensorRT variables
set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
set(TensorRT_LIBRARIES ${TensorRT_LIBRARY} ${TensorRT_PLUGIN_LIBRARY})

# Handle REQUIRED and QUIET arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_LIBRARY TensorRT_PLUGIN_LIBRARY
    VERSION_VAR TensorRT_VERSION
)

# Mark as advanced
mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_LIBRARY TensorRT_PLUGIN_LIBRARY)

# Print status
if(TensorRT_FOUND)
    message(STATUS "Found TensorRT: ${TensorRT_VERSION}")
    message(STATUS "  Include: ${TensorRT_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${TensorRT_LIBRARIES}")
else()
    message(STATUS "TensorRT not found")
    message(STATUS "  Searched in: ${TensorRT_SEARCH_PATHS}")
    message(STATUS "  Set TensorRT_DIR to specify custom location")
endif() 
