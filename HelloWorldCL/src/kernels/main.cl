#ifndef __cl__
#define __kernel
#define __global
#define __constant
#endif

//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void addVectors(__global const float *a, __global const float *b, __global float *result, const int size)
{
    int i = get_global_id(0);

    if (i < size)
    {
        result[i] = a[i] + b[i];
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
