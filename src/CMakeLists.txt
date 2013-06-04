# Test 1: Thrust partitioning
CUDA_ADD_EXECUTABLE(gpu_partitioning "${CMAKE_CURRENT_SOURCE_DIR}/gpu_partitioning.cu")
TARGET_LINK_LIBRARIES(gpu_partitioning ${CUDA_LIBRARIES} stdc++)

# Test 2: Hamming distance with Thrust
CUDA_ADD_EXECUTABLE(gpu_hamming_distance "${CMAKE_CURRENT_SOURCE_DIR}/gpu_hamming_distance.cu")
TARGET_LINK_LIBRARIES(gpu_hamming_distance ${CUDA_LIBRARIES} stdc++)

# Test 3: Hash on GPU
CUDA_ADD_EXECUTABLE(gpu_hash "${CMAKE_CURRENT_SOURCE_DIR}/gpu_hash.cu")
TARGET_LINK_LIBRARIES(gpu_hash ${CUDA_LIBRARIES} stdc++)

# Test 4: Global vs shared pointer test
CUDA_ADD_EXECUTABLE(gpu_pointer_type "${CMAKE_CURRENT_SOURCE_DIR}/gpu_pointer_type.cu")
TARGET_LINK_LIBRARIES(gpu_pointer_type ${CUDA_LIBRARIES} stdc++)

# Test 5: Global memory vs constant memory vs texture memory test
CUDA_ADD_EXECUTABLE(gpu_memory_comparison "${CMAKE_CURRENT_SOURCE_DIR}/gpu_memory_comparison.cu")
TARGET_LINK_LIBRARIES(gpu_memory_comparison ${CUDA_LIBRARIES} stdc++)

# Test 6: Element-wise product of complex vectors
CUDA_ADD_EXECUTABLE(gpu_complex_multiplication "${CMAKE_CURRENT_SOURCE_DIR}/gpu_complex_multiplication.cu")
TARGET_LINK_LIBRARIES(gpu_complex_multiplication ${CUDA_LIBRARIES} stdc++)
