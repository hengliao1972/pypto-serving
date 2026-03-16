cmake_minimum_required(VERSION 3.16)
project(pypto_serving LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

# ---- Locate pre-built pypto_runtime_distributed libraries ----
set(LINQU_RUNTIME_DIR ${CMAKE_SOURCE_DIR}/../pypto_runtime_distributed)
set(LINQU_BUILD_DIR ${LINQU_RUNTIME_DIR}/build)

# Headers from both projects
include_directories(${LINQU_RUNTIME_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Import pre-built static libraries
foreach(LIB linqu_runtime_lib linqu_ring linqu_core linqu_profiling linqu_discovery linqu_transport)
    add_library(${LIB} STATIC IMPORTED)
    set_target_properties(${LIB} PROPERTIES
        IMPORTED_LOCATION ${LINQU_BUILD_DIR}/lib${LIB}.a
    )
endforeach()

# ---- pypto_serving library (engine + stubs) ----
file(GLOB_RECURSE SERVING_SRC
    src/l2_stubs/*.cpp
    src/engine/*.cpp
    src/kv/*.cpp
)

add_library(pypto_serving_lib STATIC ${SERVING_SRC})
target_link_libraries(pypto_serving_lib
    linqu_runtime_lib
    linqu_ring
    linqu_core
    linqu_profiling
    linqu_discovery
    linqu_transport
    pthread
    dl
)

# ---- TestPath shared library (for Python ctypes) ----
add_library(pypto_testpath SHARED src/frontend/test_path.cpp)
target_link_libraries(pypto_testpath pypto_serving_lib)
set_target_properties(pypto_testpath PROPERTIES
    PREFIX ""
    OUTPUT_NAME "pypto_testpath"
)

# ---- Tests ----
enable_testing()

macro(serving_add_test TEST_NAME TEST_SRC)
    add_executable(${TEST_NAME} ${TEST_SRC})
    target_link_libraries(${TEST_NAME} pypto_serving_lib)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endmacro()

serving_add_test(test_phase0_smoke tests/test_phase0_smoke.cpp)

# Phase 1: C++ integration test using TestPath C API
add_executable(test_phase1_testpath tests/test_phase1_testpath.cpp src/frontend/test_path.cpp)
target_link_libraries(test_phase1_testpath pypto_serving_lib)
add_test(NAME test_phase1_testpath COMMAND test_phase1_testpath)

# Phase 2: Radix Tree & KV Cache unit tests
serving_add_test(test_phase2_radix_kv tests/test_phase2_radix_kv.cpp)

# Phase 3: Serve request integration tests
serving_add_test(test_phase3_serve tests/test_phase3_serve.cpp)
