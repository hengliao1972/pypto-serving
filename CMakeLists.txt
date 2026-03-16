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

# ---- pypto_serving library ----
file(GLOB_RECURSE SERVING_SRC
    src/l2_stubs/*.cpp
    src/engine/*.cpp
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

# ---- Tests ----
enable_testing()

macro(serving_add_test TEST_NAME TEST_SRC)
    add_executable(${TEST_NAME} ${TEST_SRC})
    target_link_libraries(${TEST_NAME} pypto_serving_lib)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endmacro()

serving_add_test(test_phase0_smoke tests/test_phase0_smoke.cpp)
