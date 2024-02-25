#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <cufft.h>
#include <windows.h>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <functional>
#include <algorithm>
#include <chrono>

#ifdef min
#undef min
#undef max
#endif

#include "device.h"
#define NAMESPACE_BEGIN(name) \
    namespace name            \
    {
#define NAMESPACE_END(name) }

NAMESPACE_BEGIN(cu);
// #define XMP_IMAD
//  #define XMP_XMAD
//  #define XMP_WMAD
//--------------------------------------------------------------------------------------------------------------------//
typedef char s8;
typedef unsigned char u8;
typedef const u8 cu8;

typedef signed short s16;
typedef unsigned short u16;

typedef unsigned int ui32;

typedef signed long s32;
typedef unsigned long u32;
typedef signed long long s64;
typedef unsigned long long u64;
typedef s64 l64;

typedef float f32;
typedef double f64;
//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
const u32 BLOCK = 256;
//--------------------------------------------------------------------------------------------------------------------//
__host__ __device__ __forceinline__ u32 DIV(u32 count, u32 block)
{
    return (count + block - 1) / block;
}
//--------------------------------------------------------------------------------------------------------------------//
__host__ __device__ __forceinline__ u32 GRID(u32 count, u32 block = BLOCK)
{
    return DIV(count, block);
}
//--------------------------------------------------------------------------------------------------------------------//
__host__ __device__ __forceinline__ unsigned int nextPowerOf2(unsigned int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}
//--------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(cu);