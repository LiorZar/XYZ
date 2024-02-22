cmake_minimum_required(VERSION 3.20)
project(PlotX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Ensure 64-bit build, adjust flags depending on compiler and platform
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)   
    message(FATAL_ERROR "Target must be 64 bits")
endif()

file(GLOB_RECURSE SOURCES "src/*.cpp") 
file(GLOB_RECURSE HEADERS "src/*.h" "src/*.hpp")

add_executable(PlotX ${SOURCES} ${HEADERS})

set(GLEW_DIR $ENV{GLEW_DIR})
set(GLEW_INCLUDE_DIR "${GLEW_DIR}/include")
set(GLEW_LIBRARY "${GLEW_DIR}/lib/Release/x64/glew32.lib")

message(STATUS "GLEW_DIR: ${GLEW_DIR}")
message(STATUS "GLEW_LIBRARY: ${GLEW_LIBRARY}")
message(STATUS "GLEW_INCLUDE_DIR: ${GLEW_INCLUDE_DIR}")

set(GLFW_DIR $ENV{GLFW_DIR})
set(GLFW_INCLUDE_DIR "${GLFW_DIR}/include")
set(GLFW_LIBRARY "${GLFW_DIR}/lib-mingw-w64/glfw3.dll")

set(CLFFT_INCLUDE_DIR $ENV{CLFFT_INCLUDE_DIR})
set(CLFFT_LIBRARY $ENV{CLFFT_LIBRARY}/clFFT.lib)

find_package(OpenCL 2.1 REQUIRED)
find_package(OpenGL REQUIRED)
# find_package(GLFW3 REQUIRED)
# find_package(GLEW REQUIRED)
# find_package(glm REQUIRED)


include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR} ${CLFFT_INCLUDE_DIR})


target_include_directories(PlotX PRIVATE ${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR} ${OpenCL_INCLUDE_DIRS} ${CLFFT_INCLUDE_DIR} )
target_link_libraries(PlotX OpenGL::GL ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OpenCL_LIBRARIES} ${CLFFT_LIBRARY}) # Use glfw3 for static linking

