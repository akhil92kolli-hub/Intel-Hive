cmake_minimum_required(VERSION 3.22.1)
project(intelhive_jni LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
  message(FATAL_ERROR "IntelHive M1 supports arm64-v8a only")
endif()

if(NOT DEFINED INTELHIVE_LLAMA_SOURCE_DIR OR NOT EXISTS "${INTELHIVE_LLAMA_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Set intelhive.llamaSourceDir to the pinned llama.cpp source directory")
endif()

set(INTELHIVE_LAYER_RANGE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../native/layer-range")
file(STRINGS "${INTELHIVE_LAYER_RANGE_DIR}/upstream.lock" LLAMA_REVISION_LINE REGEX "^revision=")
string(REPLACE "revision=" "" LLAMA_PINNED_REVISION "${LLAMA_REVISION_LINE}")
string(LENGTH "${LLAMA_PINNED_REVISION}" LLAMA_PINNED_REVISION_LENGTH)
if(NOT LLAMA_PINNED_REVISION_LENGTH EQUAL 40 OR
    NOT LLAMA_PINNED_REVISION MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR "The native layer-range upstream.lock does not contain a pinned llama.cpp revision")
endif()
if(EXISTS "${INTELHIVE_LLAMA_SOURCE_DIR}/.git")
  execute_process(
    COMMAND git -C "${INTELHIVE_LLAMA_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE LLAMA_ACTUAL_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE LLAMA_GIT_RESULT
  )
  if(NOT LLAMA_GIT_RESULT EQUAL 0 OR NOT LLAMA_ACTUAL_REVISION STREQUAL LLAMA_PINNED_REVISION)
    message(FATAL_ERROR "llama.cpp revision mismatch: expected ${LLAMA_PINNED_REVISION}, got ${LLAMA_ACTUAL_REVISION}")
  endif()
else()
  file(STRINGS "${INTELHIVE_LLAMA_SOURCE_DIR}/CMakeLists.txt" LLAMA_VERSION_LINES
    REGEX "^set\\(LLAMA_VERSION_(MAJOR|MINOR|PATCH) ")
  if(NOT LLAMA_VERSION_LINES MATCHES "LLAMA_VERSION_MAJOR 0" OR
      NOT LLAMA_VERSION_LINES MATCHES "LLAMA_VERSION_MINOR 5" OR
      NOT LLAMA_VERSION_LINES MATCHES "LLAMA_VERSION_PATCH 0")
    message(FATAL_ERROR "Extracted llama.cpp source must be version 0.5.0 (${LLAMA_PINNED_REVISION})")
  endif()
  file(SHA256 "${INTELHIVE_LLAMA_SOURCE_DIR}/src/models/qwen2.cpp" LLAMA_QWEN2_SOURCE_SHA256)
  if(NOT LLAMA_QWEN2_SOURCE_SHA256 STREQUAL "0da43ff61f449be3a3da8ca2e7cc0eee119b4b8e07c8accce0041998241824e8")
    message(FATAL_ERROR "Extracted llama.cpp Qwen2 source does not match pinned revision ${LLAMA_PINNED_REVISION}")
  endif()
endif()

# Build the same source revision and CPU backend used by the host layer-range test.
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
add_subdirectory("${INTELHIVE_LLAMA_SOURCE_DIR}" llama.cpp)

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
