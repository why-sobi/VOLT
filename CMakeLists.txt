cmake_minimum_required(VERSION 3.10)

project(VOLT VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(VOLT src/main.cpp)

target_include_directories(VOLT PUBLIC include)
# 3rd party includes

target_include_directories(VOLT SYSTEM PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/3rd party/eigen"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rd party/rapidcsv"
)

# 1. Find OpenMP package
find_package(OpenMP REQUIRED)

# 2. Link OpenMP to your target
if(OpenMP_CXX_FOUND)
    target_link_libraries(VOLT PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    # This is the MSVC version of "optimize for this specific CPU"
    target_compile_options(VOLT PRIVATE /favor:INTEL64) # or /favor:AMD64
else()
    # If you ever switch to GCC/Clang (Linux/Mac)
    target_compile_options(VOLT PRIVATE -march=native)
endif()

# 3. Optimization: Tell Eigen to use OpenMP
target_compile_definitions(VOLT PRIVATE EIGEN_USE_THREADS)