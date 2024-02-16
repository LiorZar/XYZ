cmake_minimum_required(VERSION 3.20)
project(HelloWorldCL)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Ensure 64-bit build, adjust flags depending on compiler and platform
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "Target is 64 bits")
    if(MSVC) # For MSVC, use the /arch flag if necessary
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:x64")
    else() # For GCC and Clang, the -m64 flag enforces 64-bit builds
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
    endif()
else()
    message(FATAL_ERROR "Target must be 64 bits")
endif()

find_package(OpenCL 2.1 REQUIRED)

file(GLOB SOURCES "src/*.cpp")

add_executable(HelloWorldCL ${SOURCES})
target_include_directories(HelloWorldCL PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(HelloWorldCL PRIVATE ${OpenCL_LIBRARIES})

