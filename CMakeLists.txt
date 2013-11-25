# Copyright 2013, Benjamin Chretien, LIRMM-CNRS

# Requires at least CMake 2.8 to find CUDA
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
ENABLE_LANGUAGE(CXX)

INCLUDE(cmake/base.cmake)

# Searching for CUDA
FIND_PACKAGE(CUDA REQUIRED)

SET(PROJECT_NAME gpu-sandbox)
SET(PROJECT_DESCRIPTION "GPU Sandbox")
SET(PROJECT_URL "https://github.com/bchretien/GPU-sandbox")

SET(HEADERS "")

INCLUDE_DIRECTORIES(
  SYSTEM
  "${CMAKE_CURRENT_SOURCE_DIR}/thrust"
)

# FIXME: path may be wrong, but is not directly available with FindCUDA.cmake
LINK_DIRECTORIES (
  "/opt/cuda/lib64"
)

SETUP_PROJECT()

IF(CUDA_FOUND)
  LIST(APPEND MG_CUDA_LIBRARIES ${CUDA_LIBRARIES} -lcutil)

  # Disable warnings: FindCUDA.cmake does not handle system includes correctly
  SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-w")

  # Error with GCC 4.7: https://bugs.archlinux.org/task/29359
  EXECUTE_PROCESS(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
  IF (GCC_VERSION VERSION_GREATER 4.7.1)
    MESSAGE("--- GCC 4.7.1 or higher detected, using pre-include to solve compilation error")
    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; --pre-include ${CMAKE_CURRENT_SOURCE_DIR}/include/undef_atomics_int128.h")
  ENDIF()

  # Choose architecture
  SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30")

  IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
      # -G -g : CUDA debug mode (useful with cuda-memcheck)
      SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-G;-g;-lineinfo")
      # -DTHRUST_DEBUG : Thrust debug mode
      #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-DTHRUST_DEBUG;-lineinfo")
  ELSE()
      SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcicc;-O3;-Xptxas;-O3;-g;-lineinfo")
  ENDIF (CMAKE_BUILD_TYPE STREQUAL "Debug")
ENDIF(CUDA_FOUND)

# Set output
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE})
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})

ADD_SUBDIRECTORY(src)
