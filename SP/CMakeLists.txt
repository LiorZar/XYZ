cmake_minimum_required(VERSION 3.18)
project(SP)


# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


if(UNIX) 
    message("UNIX")
    set(clFFT_DIR "home/${USER}/Documents/halbana/clFFT-master/src/build")
    list( APPEND CMAKE_PREFIX_PATH "/usr/local/lib64/cmake/clFFT")
    find_package(clFFT REQUIRED)
endif()

if(WIN32)
    message("WIN32")
    set(clFFT_DIR $ENV{CLFFT_INCLUDE_DIR})
    set(CLFFT_INCLUDE_DIRS $ENV{CLFFT_INCLUDE_DIR})
    set(CLFFT_LIBRARIES $ENV{CLFFT_LIBRARY}/clFFT.lib)
    include_directories(${CLFFT_INCLUDE_DIR})
    link_directories(${CLFFT_LIBRARY})
endif()

find_package(OpenCL REQUIRED)


file(GLOB_RECURSE SOURCES "src/*.cpp") 
file(GLOB_RECURSE HEADERS "src/*.h" "src/*.hpp")

add_executable(SP ${SOURCES} ${HEADERS})
target_include_directories(SP PRIVATE ${OpenCL_INCLUDE_DIRS} ${CLFFT_INCLUDE_DIRS})
target_link_libraries(SP PRIVATE ${OpenCL_LIBRARIES} ${CLFFT_LIBRARIES})


