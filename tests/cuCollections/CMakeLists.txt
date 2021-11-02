cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project (test)

# TODO: cache it
CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY NVIDIA/cuCollections
  GIT_TAG dev
  OPTIONS
     "BUILD_TESTS OFF"
     "BUILD_BENCHMARKS OFF"
     "BUILD_EXAMPLES OFF"
)


set(cuco_includes "${cuco_SOURCE_DIR}/include")
file(TO_NATIVE_PATH ${cuco_includes} cuco_includes_native)


add_executable(test_cuco test_cuco.cu)
set_target_properties(test_cuco PROPERTIES
								FOLDER "tests")
set_target_properties(test_cuco PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHS})
target_link_libraries(test_cuco PRIVATE dsbench
                                PRIVATE cuco)


#VS
set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES ${CMAKE_VS_SDK_INCLUDE_DIRECTORIES} ${cuco_includes})
message(STATUS "CMAKE_VS_SDK_INCLUDE_DIRECTORIES: ${CMAKE_VS_SDK_INCLUDE_DIRECTORIES}")
