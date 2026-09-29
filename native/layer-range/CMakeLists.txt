cmake_minimum_required(VERSION 3.22)
project(intelhive_layer_range LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(LLAMA_SOURCE_DIR "" CACHE PATH "llama.cpp source at the revision in upstream.lock")
if(NOT EXISTS "${LLAMA_SOURCE_DIR}/ggml/CMakeLists.txt")
  message(FATAL_ERROR "Set LLAMA_SOURCE_DIR to the pinned llama.cpp checkout")
endif()
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/upstream.lock" LLAMA_REVISION_LINE REGEX "^revision=")
string(REPLACE "revision=" "" LLAMA_PINNED_REVISION "${LLAMA_REVISION_LINE}")
string(LENGTH "${LLAMA_PINNED_REVISION}" LLAMA_PINNED_REVISION_LENGTH)
if(NOT LLAMA_PINNED_REVISION_LENGTH EQUAL 40 OR
    NOT LLAMA_PINNED_REVISION MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR
    "upstream.lock must contain a 40-character llama.cpp Git revision")
endif()

if(EXISTS "${LLAMA_SOURCE_DIR}/.git")
  execute_process(
    COMMAND git -C "${LLAMA_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE LLAMA_ACTUAL_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE LLAMA_GIT_RESULT
  )
  if(NOT LLAMA_GIT_RESULT EQUAL 0 OR NOT LLAMA_ACTUAL_REVISION STREQUAL LLAMA_PINNED_REVISION)
    message(FATAL_ERROR
      "llama.cpp revision mismatch: expected ${LLAMA_PINNED_REVISION}, got ${LLAMA_ACTUAL_REVISION}")
  endif()
else()
  file(STRINGS "${LLAMA_SOURCE_DIR}/CMakeLists.txt" LLAMA_VERSION_LINES
    REGEX "^set\\(LLAMA_VERSION_(MAJOR|MINOR|PATCH) ")
  if(NOT LLAMA_VERSION_LINES MATCHES "LLAMA_VERSION_MAJOR 0" OR
      NOT LLAMA_VERSION_LINES MATCHES "LLAMA_VERSION_MINOR 5" OR
      NOT LLAMA_VERSION_LINES MATCHES "LLAMA_VERSION_PATCH 0")
    message(FATAL_ERROR
      "Cannot verify extracted llama.cpp sources: expected version 0.5.0 for ${LLAMA_PINNED_REVISION}")
  endif()
  file(SHA256 "${LLAMA_SOURCE_DIR}/src/models/qwen2.cpp" LLAMA_QWEN2_SOURCE_SHA256)
  if(NOT LLAMA_QWEN2_SOURCE_SHA256 STREQUAL "0da43ff61f449be3a3da8ca2e7cc0eee119b4b8e07c8accce0041998241824e8")
    message(FATAL_ERROR
      "Extracted llama.cpp Qwen2 source checksum does not match revision ${LLAMA_PINNED_REVISION}")
  endif()
endif()
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_SOURCE_DIR}" llama.cpp)
add_library(intelhive_layer_range STATIC layer_range.cpp qwen2_gguf.cpp)
target_include_directories(intelhive_layer_range PUBLIC .)
target_link_libraries(intelhive_layer_range PUBLIC ggml ggml-cpu)
add_executable(layer_range_test layer_range_test.cpp)
target_link_libraries(layer_range_test PRIVATE intelhive_layer_range)
enable_testing()
add_test(NAME layer_range_equivalence COMMAND layer_range_test)
add_executable(qwen2_gguf_test qwen2_gguf_test.cpp)
target_link_libraries(qwen2_gguf_test PRIVATE intelhive_layer_range llama)
set(INTELHIVE_QWEN_GGUF "" CACHE FILEPATH "Local Qwen2.5-3B-Instruct GGUF for real-weight tests")
if(INTELHIVE_QWEN_GGUF)
  add_test(NAME qwen2_gguf_layer_range COMMAND qwen2_gguf_test "${INTELHIVE_QWEN_GGUF}")
endif()
