cmake_minimum_required(VERSION 3.20)
project(HelloWorldCL)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Ensure 64-bit build, adjust flags depending on compiler and platform
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)   
    message(FATAL_ERROR "Target must be 64 bits")
endif()

find_package(OpenCL 2.1 REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "src/*.h" "src/*.hpp")

add_executable(HelloWorldCL ${SOURCES} ${HEADERS})
target_include_directories(HelloWorldCL PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(HelloWorldCL PRIVATE ${OpenCL_LIBRARIES})


