cmake_minimum_required(VERSION 3.20)
project(SP)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Ensure 64-bit build, adjust flags depending on compiler and platform
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)   
    message(FATAL_ERROR "Target must be 64 bits")
endif()


set(CLFFT_INCLUDE_DIR $ENV{CLFFT_INCLUDE_DIR})
set(CLFFT_LIBRARY $ENV{CLFFT_LIBRARY}\\clFFT.lib)
# set(CLFFT_LIBRARY $ENV{CLFFT_LIBRARY})

find_package(OpenCL 2.1 REQUIRED)

include_directories(${CLFFT_INCLUDE_DIR})
link_directories(${CLFFT_LIBRARY})


file(GLOB_RECURSE SOURCES "src/*.cpp") 
file(GLOB_RECURSE HEADERS "src/*.h" "src/*.hpp")

add_executable(SP ${SOURCES} ${HEADERS})
target_include_directories(SP PRIVATE ${OpenCL_INCLUDE_DIRS} ${CLFFT_INCLUDE_DIR})
target_link_libraries(SP PRIVATE ${OpenCL_LIBRARIES} ${CLFFT_LIBRARY})


