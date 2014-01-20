# Test 1: Thrust partitioning
CUDA_ADD_EXECUTABLE(gpu_partitioning "${CMAKE_CURRENT_SOURCE_DIR}/gpu_partitioning.cu")

# Test 2: Hamming distance with Thrust
CUDA_ADD_EXECUTABLE(gpu_hamming_distance "${CMAKE_CURRENT_SOURCE_DIR}/gpu_hamming_distance.cu")

# Test 3: Hash on GPU
CUDA_ADD_EXECUTABLE(gpu_hash "${CMAKE_CURRENT_SOURCE_DIR}/gpu_hash.cu")

# Test 4: Global vs shared pointer test
CUDA_ADD_EXECUTABLE(gpu_pointer_type "${CMAKE_CURRENT_SOURCE_DIR}/gpu_pointer_type.cu")

# Test 5: Global memory vs constant memory vs texture memory test
CUDA_ADD_EXECUTABLE(gpu_memory_comparison "${CMAKE_CURRENT_SOURCE_DIR}/gpu_memory_comparison.cu")

# Test 6: Element-wise product of complex vectors
CUDA_ADD_EXECUTABLE(gpu_complex_multiplication "${CMAKE_CURRENT_SOURCE_DIR}/gpu_complex_multiplication.cu")

# Test 7: Count the occurrences of some specific case
CUDA_ADD_EXECUTABLE(gpu_counters "${CMAKE_CURRENT_SOURCE_DIR}/gpu_counters.cu")

# Test 8: Local register passing
CUDA_ADD_EXECUTABLE(gpu_register_passing "${CMAKE_CURRENT_SOURCE_DIR}/gpu_register_passing.cu")

# Test 9: Signal handling
CUDA_ADD_EXECUTABLE(gpu_sigint "${CMAKE_CURRENT_SOURCE_DIR}/gpu_sigint.cu")

# Test 10: scan operation on multi-dimensional arrays
CUDA_ADD_EXECUTABLE(gpu_array_scan "${CMAKE_CURRENT_SOURCE_DIR}/gpu_array_scan.cu")

# Test 11: NVTX markers
CUDA_ADD_EXECUTABLE(gpu_nvtx "${CMAKE_CURRENT_SOURCE_DIR}/gpu_nvtx.cu")
TARGET_LINK_LIBRARIES(gpu_nvtx ${CUDA_LIBRARIES} nvToolsExt)

# Test 12: array of structures of pointers
CUDA_ADD_EXECUTABLE(gpu_structure_pointers "${CMAKE_CURRENT_SOURCE_DIR}/gpu_structure_pointers.cu")

# Test 13: using Boost MPL in CUDA kernel
CUDA_ADD_EXECUTABLE(gpu_boost_mpl "${CMAKE_CURRENT_SOURCE_DIR}/gpu_boost_mpl.cu")

# Test 14: redirect GPU printf to anothe stream
CUDA_ADD_EXECUTABLE(gpu_printf_redirect "${CMAKE_CURRENT_SOURCE_DIR}/gpu_printf_redirect.cu")

# Bug 1: compiler crash with wrong array reference
CUDA_ADD_EXECUTABLE(gpu_bug_1 "${CMAKE_CURRENT_SOURCE_DIR}/cuda_bug/gpu_nvcc_ice.cu")

# Bug 2: compiler crash
CUDA_ADD_EXECUTABLE(gpu_bug_2 "${CMAKE_CURRENT_SOURCE_DIR}/cuda_bug/gpu_nvcc_ice2.cu")

# Bug 3: assert compiler error
CUDA_ADD_EXECUTABLE(gpu_bug_3 "${CMAKE_CURRENT_SOURCE_DIR}/cuda_bug/gpu_nvcc_assert.cu")
