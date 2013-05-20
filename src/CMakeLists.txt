# Test 1: Thrust partitioning
CUDA_ADD_EXECUTABLE(gpu_partitioning "${CMAKE_CURRENT_SOURCE_DIR}/gpu_partitioning.cu")
TARGET_LINK_LIBRARIES(gpu_partitioning ${CUDA_LIBRARIES} stdc++)

# Test 2: Hamming distance with Thrust
CUDA_ADD_EXECUTABLE(gpu_hamming_distance "${CMAKE_CURRENT_SOURCE_DIR}/gpu_hamming_distance.cu")
TARGET_LINK_LIBRARIES(gpu_hamming_distance ${CUDA_LIBRARIES} stdc++)

# Test 3: Hash on GPU
CUDA_ADD_EXECUTABLE(gpu_hash "${CMAKE_CURRENT_SOURCE_DIR}/gpu_hash.cu")
TARGET_LINK_LIBRARIES(gpu_hash ${CUDA_LIBRARIES} stdc++)
