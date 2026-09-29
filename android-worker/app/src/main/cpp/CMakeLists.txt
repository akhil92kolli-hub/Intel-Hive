cmake_minimum_required(VERSION 3.22.1)
project(intelhive_jni LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
  message(FATAL_ERROR "IntelHive M1 supports arm64-v8a only")
endif()

if(NOT DEFINED INTELHIVE_LLAMA_SOURCE_DIR OR NOT EXISTS "${INTELHIVE_LLAMA_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Set intelhive.llamaSourceDir to the pinned llama.cpp source directory")
endif()

# Build the same source revision used by the host layer-range feasibility test.
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
add_subdirectory("${INTELHIVE_LLAMA_SOURCE_DIR}" llama.cpp)

set(INTELHIVE_LAYER_RANGE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../native/layer-range")
add_library(intelhive_layer_range STATIC
  "${INTELHIVE_LAYER_RANGE_DIR}/layer_range.cpp"
  "${INTELHIVE_LAYER_RANGE_DIR}/qwen2_gguf.cpp")
target_include_directories(intelhive_layer_range PUBLIC
  "${INTELHIVE_LAYER_RANGE_DIR}"
  "${INTELHIVE_LLAMA_SOURCE_DIR}/ggml/include"
  "${INTELHIVE_LLAMA_SOURCE_DIR}/ggml/src")
target_link_libraries(intelhive_layer_range PUBLIC ggml ggml-cpu)

add_library(intelhive_jni SHARED intelhive_jni.cpp)
target_link_libraries(intelhive_jni PRIVATE intelhive_layer_range llama android log)
