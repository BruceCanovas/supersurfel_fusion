find_package( CUDA REQUIRED )

# Auto-detect the CUDA compute capability.
SET(CMAKE_MODULE_PATH "${supersurfel_fusion_SOURCE_DIR}/cmake")
IF(NOT DEFINED CUDA_COMPUTE_CAPABILITY)
  INCLUDE("${CMAKE_MODULE_PATH}/CUDACheckCompute.cmake")
ENDIF()

# Set the compute capability flags.
FOREACH(compute_capability ${CUDA_COMPUTE_CAPABILITY})
  LIST(APPEND CUDA_NVCC_FLAGS --generate-code;arch=compute_${compute_capability},code=compute_${compute_capability})
  LIST(APPEND CUDA_NVCC_FLAGS --generate-code;arch=compute_${compute_capability},code=sm_${compute_capability})
ENDFOREACH()

LIST(APPEND CUDA_NVCC_FLAGS --use_fast_math)

# Make sure that C++11 support is enabled when compiling with nvcc. From CMake 3.5 onwards,
# the host flag -std=c++11 is automatically propagated to nvcc. Manually setting it prevents
# the project from building.
IF(${CMAKE_VERSION} VERSION_LESS 3.5)
  SET(CUDA_NVCC_FLAGS -std=c++11; ${CUDA_NVCC_FLAGS})
ENDIF()

