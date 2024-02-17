#ifndef __cl__
#define __kernel
#define __global
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
float2 getSample(const float2 *data, int channel, int sample, int numChannels)
{
    return data[sample * numChannels + channel];
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void Transpose1D(__global const float *input, __global float *output, const int cols, const int rows)
{
    int idx = get_global_id(0);
    if (idx >= cols * rows)
        return;
    int row = idx / cols;
    int col = idx % cols;

    output[col * rows + row] = input[row * cols + col];
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void Transpose2D(__global const float2 *input, __global float2 *output, const int cols, const int rows)
{
    int idx = get_global_id(0);
    if (idx >= cols * rows)
        return;
    int row = idx / cols;
    int col = idx % cols;

    output[col * rows + row] = input[row * cols + col];
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void Transpose2DPalanar(__global const float2 *input, __global float *outputX, __global float *outputY, const int cols, const int rows)
{
    int idx = get_global_id(0);
    if (idx >= cols * rows)
        return;
    int row = idx / cols;
    int col = idx % cols;

    outputX[col * rows + row] = input[row * cols + col].x;
    outputY[col * rows + row] = input[row * cols + col].y;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void InverseTranspose2DPalanar(__global const float *inputX, __global const float *inputY, __global float2 *output, const int cols, const int rows)
{
    int idx = get_global_id(0);
    if (idx >= cols * rows)
        return;
    int row = idx / cols;
    int col = idx % cols;

    float2 smp = {inputX[row * cols + col], inputY[row * cols + col]};
    output[col * rows + row] = smp;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void convolve1D(
    __global const float *prev,
    __global const float *curr,
    __global const float *filter,
    __global float *output,
    const int inputSize,
    const int filterSize,
    const int numChannels)
{
    int idx = get_global_id(0); // sample index [0, inputSize]
    if (idx >= inputSize)
        return;

    const int sample_per_channel = inputSize / numChannels;
    const int sample = idx % sample_per_channel;
    const int channel = idx / sample_per_channel;
    const int base = channel * sample_per_channel;
    const int fbase = channel * filterSize;

    float fr;
    float smp;
    float result = 0.0f;
    for (int f = 0, s = sample + 1 - filterSize; f < filterSize; ++f, ++s)
    {
        fr = filter[fbase + f];
        smp = s < 0 ? prev[base + s + sample_per_channel] : curr[base + s];

        result += fr * smp;
    }
    output[idx] = result;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void convolve2D(
    __global const float2 *prev,
    __global const float2 *curr,
    __global const float *filter,
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
