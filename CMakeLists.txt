cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(SET CMP0104 OLD)

project(DSBench LANGUAGES CXX CUDA)

# Set c++ standard
set(CMAKE_CXX_STANDARD 17)

# Direct all output to /bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# tools
find_package(CUDAToolkit REQUIRED)
include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/AutoDetectCudaArch.cmake)


# options
option(build_benchmarks "Build benchmarks" OFF)
option(build_tests "Build tests" ON)


set(external_includes "${CMAKE_CURRENT_SOURCE_DIR}/external")
set(ptx_directory "${CMAKE_CURRENT_SOURCE_DIR}/ptx")
set(cuda_includes     ${CUDAToolkit_INCLUDE_DIRS})
set(includes_directory "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(benchmarks_directory "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks")
set(tests_directory "${CMAKE_CURRENT_SOURCE_DIR}/tests")
set(benchmarks_directory "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks")

# Native
file(TO_NATIVE_PATH ${external_includes} external_includes_native)
file(TO_NATIVE_PATH ${ptx_directory} ptx_directory_native)
file(TO_NATIVE_PATH ${cuda_includes} cuda_includes_native)
file(TO_NATIVE_PATH ${includes_directory} includes_directory_native)
file(TO_NATIVE_PATH ${benchmarks_directory} benchmarks_directory_native)
file(TO_NATIVE_PATH ${tests_directory} tests_directory_native)
file(TO_NATIVE_PATH ${benchmarks_directory} benchmarks_directory_native)



# CUDA and C++ compiler flags
set(CXX_FLAGS
        $<IF:$<CXX_COMPILER_ID:MSVC>,
        /std:c++17,
        --std=c++17
        >)
set(CUDA_FLAGS
    $<IF:$<CXX_COMPILER_ID:MSVC>,
    -Xcompiler=/std:c++17,
    -Xcompiler=-std=c++17 -Wextra>
    $<IF:$<CXX_COMPILER_ID:MSVC>,
    -Xcompiler=/W4,
    -Xcompiler=-Wall>
    $<IF:$<CONFIG:Debug>,
        -G -src-in-ptx,
        -lineinfo>
    # ${CUDA_ARCHS}
	--expt-extended-lambda
    -Wno-deprecated-gpu-targets #silence deprecated architectures
    --expt-relaxed-constexpr
    $<$<CONFIG:Release>:-Xptxas -warn-spills -res-usage>
    )


add_library(dsbench INTERFACE)

target_compile_options(dsbench INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)

target_include_directories(dsbench
    INTERFACE ${external_includes}
    INTERFACE ${cuda_includes}
    INTERFACE ${includes_directory})

target_link_libraries(dsbench
    INTERFACE CUDA::cuda_driver
    INTERFACE CUDA::cudart_static)



# visual studio setup
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT test_cuco)
set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES
    $(VC_IncludePath)
    $(WindowsSDK_IncludePath)
    ${external_includes_native}
    ${includes_directory_native}
    ${cuda_includes_native}
    )


if(build_tests)
   add_subdirectory(${tests_directory})
endif(build_tests)

if(build_benchmarks)
    add_subdirectory(${benchmarks_directory})
endif(build_benchmarks)