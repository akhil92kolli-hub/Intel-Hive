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

# Build the pinned llama.cpp revision with CPU fallback and Vulkan execution support.
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
set(GGML_VULKAN ON CACHE BOOL "" FORCE)
if(ANDROID AND (NOT Vulkan_GLSLC_EXECUTABLE OR
    NOT EXISTS "${Vulkan_GLSLC_EXECUTABLE}"))
  if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    set(INTELHIVE_NDK_HOST_TAG "darwin-x86_64")
  elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set(INTELHIVE_NDK_HOST_TAG "linux-x86_64")
  elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(INTELHIVE_NDK_HOST_TAG "windows-x86_64")
  endif()
  if(DEFINED INTELHIVE_NDK_HOST_TAG)
    set(INTELHIVE_NDK_GLSLC
      "${CMAKE_ANDROID_NDK}/shader-tools/${INTELHIVE_NDK_HOST_TAG}/glslc")
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
      set(INTELHIVE_NDK_GLSLC "${INTELHIVE_NDK_GLSLC}.exe")
    endif()
    if(EXISTS "${INTELHIVE_NDK_GLSLC}")
      set(Vulkan_GLSLC_EXECUTABLE "${INTELHIVE_NDK_GLSLC}" CACHE FILEPATH
        "glslc compiler bundled with the Android NDK" FORCE)
    endif()
  endif()
endif()
if(ANDROID AND CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND NOT DEFINED SPIRV-Headers_DIR)
  foreach(SPIRV_HEADERS_PREFIX "/opt/homebrew/opt/spirv-headers" "/usr/local/opt/spirv-headers")
    set(SPIRV_HEADERS_CONFIG_DIR "${SPIRV_HEADERS_PREFIX}/share/cmake/SPIRV-Headers")
    if(EXISTS "${SPIRV_HEADERS_CONFIG_DIR}/SPIRV-HeadersConfig.cmake")
      set(SPIRV-Headers_DIR "${SPIRV_HEADERS_CONFIG_DIR}" CACHE PATH
        "Host SPIRV-Headers CMake package used by the Android Vulkan shader build")
      break()
    endif()
  endforeach()
endif()
if(ANDROID)
  find_package(SPIRV-Headers CONFIG REQUIRED)
endif()
if(ANDROID)
  find_path(INTELHIVE_VULKAN_HPP_INCLUDE_DIR
    NAMES vulkan/vulkan.hpp
    HINTS
      /opt/homebrew/opt/vulkan-headers
      /usr/local/opt/vulkan-headers
      ENV VULKAN_SDK
    PATH_SUFFIXES include
    NO_CMAKE_FIND_ROOT_PATH
  )
  if(NOT INTELHIVE_VULKAN_HPP_INCLUDE_DIR)
    message(FATAL_ERROR
      "GGML Vulkan requires Vulkan-Headers with vulkan/vulkan.hpp; install the Vulkan-Headers package")
  endif()
endif()
set(INTELHIVE_GGML_VULKAN_CMAKE
  "${INTELHIVE_LLAMA_SOURCE_DIR}/ggml/src/ggml-vulkan/CMakeLists.txt")
file(READ "${INTELHIVE_GGML_VULKAN_CMAKE}" INTELHIVE_GGML_VULKAN_CMAKE_CONTENT)
set(INTELHIVE_GLSLC_PROBE_OLD [=[if (${glslc_error} MATCHES]=])
set(INTELHIVE_GLSLC_PROBE_FIXED
  [=[if (NOT glslc_result EQUAL 0 OR "${glslc_error}${glslc_output}" MATCHES]=])
string(FIND "${INTELHIVE_GGML_VULKAN_CMAKE_CONTENT}"
  "${INTELHIVE_GLSLC_PROBE_OLD}" INTELHIVE_GLSLC_PROBE_OLD_INDEX)
if(NOT INTELHIVE_GLSLC_PROBE_OLD_INDEX EQUAL -1)
  string(REPLACE "${INTELHIVE_GLSLC_PROBE_OLD}" "${INTELHIVE_GLSLC_PROBE_FIXED}"
    INTELHIVE_GGML_VULKAN_CMAKE_CONTENT "${INTELHIVE_GGML_VULKAN_CMAKE_CONTENT}")
  string(REPLACE "ERROR_VARIABLE glslc_error\n"
    "ERROR_VARIABLE glslc_error\n        RESULT_VARIABLE glslc_result\n"
    INTELHIVE_GGML_VULKAN_CMAKE_CONTENT "${INTELHIVE_GGML_VULKAN_CMAKE_CONTENT}")
  file(WRITE "${INTELHIVE_GGML_VULKAN_CMAKE}"
    "${INTELHIVE_GGML_VULKAN_CMAKE_CONTENT}")
elseif(NOT INTELHIVE_GGML_VULKAN_CMAKE_CONTENT MATCHES
    "glslc_result EQUAL 0 OR.*glslc_error.*glslc_output")
  message(FATAL_ERROR
    "Pinned GGML Vulkan GLSL extension probe has an unexpected format; refusing an unreliable Vulkan build")
endif()
add_subdirectory("${INTELHIVE_LLAMA_SOURCE_DIR}" llama.cpp)
if(TARGET ggml-vulkan)
  target_link_libraries(ggml-vulkan PRIVATE SPIRV-Headers::SPIRV-Headers)
  target_include_directories(ggml-vulkan SYSTEM PRIVATE
    "${INTELHIVE_VULKAN_HPP_INCLUDE_DIR}")
endif()

add_library(intelhive_layer_range STATIC
  "${INTELHIVE_LAYER_RANGE_DIR}/layer_range.cpp"
  "${INTELHIVE_LAYER_RANGE_DIR}/qwen2_gguf.cpp")
target_include_directories(intelhive_layer_range PUBLIC
  "${INTELHIVE_LAYER_RANGE_DIR}"
  "${INTELHIVE_LLAMA_SOURCE_DIR}/ggml/include"
  "${INTELHIVE_LLAMA_SOURCE_DIR}/ggml/src")
target_link_libraries(intelhive_layer_range PUBLIC ggml ggml-cpu llama)
if(GGML_VULKAN)
  target_compile_definitions(intelhive_layer_range PRIVATE INTELHIVE_HAS_VULKAN=1)
  target_link_libraries(intelhive_layer_range PUBLIC ggml-vulkan)
endif()

add_library(intelhive_jni SHARED intelhive_jni.cpp)
find_library(VULKAN_LIBRARY vulkan REQUIRED)
target_link_libraries(intelhive_jni PRIVATE intelhive_layer_range llama android log ${VULKAN_LIBRARY})
if(GGML_VULKAN)
  target_compile_definitions(intelhive_jni PRIVATE INTELHIVE_HAS_VULKAN=1)
  target_link_libraries(intelhive_jni PRIVATE ggml-vulkan)
endif()
