# https://stackoverflow.com/questions/51907755/building-a-pybind11-module-with-cpp-and-cuda-sources-using-cmake

cmake_minimum_required(VERSION 3.12)

project(gpuRIR LANGUAGES CXX CUDA)
find_package(CUDA)



# Select several CUDA architectures
include(FindCUDA)
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS Common) # Change Common by Auto for autodetect
string(REPLACE ";" " " CUDA_ARCH_FLAGS "${CUDA_ARCH_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")

set(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -D_FORCE_INLINES")
# set(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -lineinfo")

message("CALL CMakeLists.txt in root")

# need to be at least 30 or __shfl_down in reduce wont compile
IF (CUDA_VERSION VERSION_LESS "11.0")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30")
ENDIF()
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")

IF (CUDA_VERSION VERSION_GREATER "7.6")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
ENDIF()

IF ((CUDA_VERSION VERSION_GREATER "9.0") OR (CUDA_VERSION VERSION_EQUAL "9.0"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
ENDIF()

IF ((CUDA_VERSION VERSION_GREATER "10.0") OR (CUDA_VERSION VERSION_EQUAL "10.0"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
ENDIF()

IF ((CUDA_VERSION VERSION_GREATER "11.0") OR (CUDA_VERSION VERSION_EQUAL "11.0"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
ENDIF()

message(${CUDA_NVCC_FLAGS})


message("CMAKE_CUDA_FLAGS : " ${CMAKE_CUDA_FLAGS})
set(CMAKE_CUDA_FLAGS " -gencode arch=compute_80,code=sm_80")

add_library(gpuRIRcu STATIC
    src/gpuRIR_cuda.cu
)
#target_compile_options(gpuRIRcu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)


set_target_properties(gpuRIRcu PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    # CUDA_SEPARABLE_COMPILATION ON
)

add_library(gpuRIR_bind MODULE
    src/python_bind.cpp
)


set(PYBIND11_CPP_STANDARD -std=c++11)
add_subdirectory(third_party/pybind11)


target_link_libraries(gpuRIR_bind PRIVATE Threads::Threads)

set_target_properties(gpuRIR_bind PROPERTIES 
    CXX_VISIBILITY_PRESET "hidden"
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Optional IPO. Do not use IPO if it's not supported by compiler.
include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
    set_target_properties(gpuRIR_bind PROPERTIES 
        INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "IPO is not supported: ${output}")
endif()


target_link_libraries(gpuRIR_bind PRIVATE ${CUDA_curand_LIBRARY})
target_link_libraries(gpuRIR_bind PRIVATE ${CUDA_CUFFT_LIBRARIES})
target_link_libraries(gpuRIR_bind PRIVATE gpuRIRcu)
target_link_libraries(gpuRIR_bind PRIVATE pybind11::module)
