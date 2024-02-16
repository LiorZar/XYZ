#ifndef __cl__
#define __kernel
#define __global
#define __constant
typedef struct _float2
{
    float x;
    float y;
} float2;
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
float2 getSample(__constant float2 *data, int channel, int sample, int numChannels)
{
    return data[sample * numChannels + channel];
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void convolve(
    __constant float2 *prev,
    __constant float2 *curr,
    __constant float *filter,
    __global float2 *output,
    const int inputSize,
    const int filterSize,
    const int numChannels)
{
    int idx = get_global_id(0); // sample index [0, inputSize]
    if (idx >= inputSize)
        return;

    const int sample_per_channel = inputSize / numChannels;
    const int sample = idx / numChannels;
    const int channel = idx % numChannels;

    float fr;
    float2 smp;
    float2 result = {0.0f, 0.0f};
    for (int f = 0, s = sample + 1 - filterSize; f < filterSize; ++f, ++s)
    {
        fr = filter[f * numChannels + channel];
        smp = s < 0 ? getSample(prev, channel, s + sample_per_channel, numChannels) : getSample(curr, channel, s, numChannels);

        result.x += fr * smp.x;
        result.y += fr * smp.y;
    }
    output[idx] = result;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
