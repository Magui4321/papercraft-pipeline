cmake_minimum_required(VERSION 3.20)
project(papercraft VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -DNDEBUG -funroll-loops)
else()
    add_compile_options(-g -O0)
endif()

find_package(OpenMesh REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP not found - parallel unfolding disabled")
endif()

include(FetchContent)
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(stb)

FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.4.0
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(Catch2)

FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG v2.4.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(libigl)

file(GLOB SOURCES "src/*.cpp")

add_executable(papercraft ${SOURCES})

target_include_directories(papercraft PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${stb_SOURCE_DIR}
)

# Suppress warnings for OpenMesh system headers
target_include_directories(papercraft SYSTEM PRIVATE
    ${OPENMESH_INCLUDE_DIRS}
)

target_link_libraries(papercraft PRIVATE
    OpenMeshCore
    OpenMeshTools
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    igl::core
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(papercraft PRIVATE OpenMP::OpenMP_CXX)
endif()

# ── Test executable ──────────────────────────────────────────────────────────
# Collect all src/*.cpp EXCEPT main.cpp for the test library
file(GLOB LIB_SOURCES "src/*.cpp")
list(FILTER LIB_SOURCES EXCLUDE REGEX "main\\.cpp$")

add_executable(test_papercraft tests/test_math.cpp ${LIB_SOURCES})

target_include_directories(test_papercraft PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${stb_SOURCE_DIR}
)

target_include_directories(test_papercraft SYSTEM PRIVATE
    ${OPENMESH_INCLUDE_DIRS}
)

target_link_libraries(test_papercraft PRIVATE
    OpenMeshCore
    OpenMeshTools
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    Catch2::Catch2WithMain
    igl::core
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(test_papercraft PRIVATE OpenMP::OpenMP_CXX)
endif()

include(CTest)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(test_papercraft)

install(TARGETS papercraft DESTINATION bin)

message(STATUS "=== Papercraft Pipeline Configuration ===")
message(STATUS "  OpenMesh:    found")
message(STATUS "  Eigen3:      ${EIGEN3_VERSION}")
message(STATUS "  OpenMP:      ${OpenMP_CXX_FOUND}")
message(STATUS "  Build type:  ${CMAKE_BUILD_TYPE}")
message(STATUS "=========================================")
