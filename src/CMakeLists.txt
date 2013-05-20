# Test 1: Thrust partitioning
CUDA_ADD_EXECUTABLE(gpu_partitioning "${CMAKE_CURRENT_SOURCE_DIR}/gpu_partitioning.cu")
TARGET_LINK_LIBRARIES(gpu_partitioning ${CUDA_LIBRARIES} stdc++)
