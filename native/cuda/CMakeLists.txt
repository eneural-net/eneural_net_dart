cmake_minimum_required(VERSION 3.18)
project(eneural_cuda CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Detect GPU architecture automatically if not specified. Covers Volta (70)
# through Hopper/Ada (90); extend as needed for newer devices.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

# Output name carries the host arch so it can sit next to the CPU/Metal libs:
#   libeneural_cuda_x86_64.dll  (Windows) / .so (Linux)
#   libeneural_cuda_arm64.so    (Linux/Jetson)
if(NOT DEFINED ENN_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(ENN_ARCH "arm64")
    else()
        set(ENN_ARCH "x86_64")
    endif()
endif()

add_library(eneural_cuda SHARED EnnCudaBackend.cu)

set_target_properties(eneural_cuda PROPERTIES
    OUTPUT_NAME "eneural_cuda_${ENN_ARCH}"
    PREFIX "lib"
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(eneural_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# cuBLAS is required for the three per-layer GEMMs.
find_package(CUDAToolkit REQUIRED)
target_link_libraries(eneural_cuda PRIVATE CUDA::cublas)

target_compile_options(eneural_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
