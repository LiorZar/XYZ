#pragma once

#ifdef _WIN32
#define CL_TARGET_OPENCL_VERSION 210
#define CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#else
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 210
#include <CL/opencl.hpp>
#endif

#define TIMING

#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <exception>

#include <complex>
#include <cmath>
#include <algorithm>
