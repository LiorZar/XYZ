#ifndef __cl__
#define __kernel
#define __global
typedef struct _float2
{
    float x;
    float y;
} float2;

int get_global_id(int);
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
float getSample1D(__global const float *data, int channel, int sample, int numChannels)
{
    return data[sample * numChannels + channel];
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
float2 getSample2D(__global const float2 *data, int channel, int sample, int numChannels)
{
    return data[sample * numChannels + channel];
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void Transpose1D(__global const float *input, __global float *output, const int cols, const int rows, const int new_row_stride)
{
    int idx = get_global_id(0);
    if (idx >= cols * rows)
        return;
    int row = idx / cols;
    int col = idx % cols;

    output[col * new_row_stride + row] = input[row * cols + col];
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void Transpose1DC(__global const float *input, __global float2 *output, const int cols, const int rows, const int new_row_stride, const int reverse)
{
    int idx = get_global_id(0);
    if (idx >= cols * rows)
        return;
    int row = idx / cols;
    int col = idx % cols;
    int nrow = 0 == reverse ? row : rows - 1 - row;
    float2 v = {input[row * cols + col], 0.0f};
    output[col * new_row_stride + nrow] = v;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void Transpose2D(__global const float2 *input, __global float2 *output, const int cols, const int rows, const int new_row_stride, const int offset)
{
    // cols = 20, rows = 20,000, new_row = 21,600
    int idx = get_global_id(0);
    if (idx >= cols * rows) // 400,000
        return;
    int row = idx / cols; // [0,20,000)
    int col = idx % cols; // [0,20)

    output[col * new_row_stride + row + offset] = input[row * cols + col];
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void Transpose2D_copy_Prev(__global const float2 *prev, __global float2 *output, const int cols, const int rows, const int prevRows, const int new_row_stride)
{
    // cols = 20, rows = 20,000, prevRows = 500, new_row = 21,600
    int idx = get_global_id(0);
    if (idx >= cols * prevRows) // 10,000
        return;
    int row = idx / cols;                // [0,500)
    int col = idx % cols;                // [0,20)
    int prevRow = rows - prevRows + row; // [19500,20000)

    output[col * new_row_stride + row] = prev[prevRow * cols + col];
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void InverseTranspose2D(__global const float2 *input, __global float2 *output, const int cols, const int rows, const int new_row_stride, const int offset)
{
    // cols = 20, rows = 20,000, new_row = 21,600
    int idx = get_global_id(0);
    if (idx >= cols * rows) // 400,000
        return;
    int row = idx / cols; // [0,20,000)
    int col = idx % cols; // [0,20)

    output[col * rows + row] = input[row * new_row_stride + col + offset];
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void Transpose2DPalanar(__global const float2 *input, __global float *outputX, __global float *outputY, const int cols, const int rows, const int new_row_stride)
{
    int idx = get_global_id(0);
    if (idx >= cols * rows)
        return;
    int row = idx / cols;
    int col = idx % cols;

    outputX[col * new_row_stride + row] = input[row * cols + col].x;
    outputY[col * new_row_stride + row] = input[row * cols + col].y;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void Transpose2DPalanarC(__global const float2 *input, __global float2 *outputX, __global float2 *outputY, const int cols, const int rows, const int new_row_stride)
{
    int idx = get_global_id(0);
    if (idx >= cols * rows)
        return;
    int row = idx / cols;
    int col = idx % cols;

    float2 x = {input[row * cols + col].x, 0.0f};
    float2 y = {input[row * cols + col].y, 0.0f};
    outputX[col * new_row_stride + row] = x;
    outputY[col * new_row_stride + row] = y;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void InverseTranspose2DPalanar(__global const float *inputX, __global const float *inputY, __global float2 *output, const int cols, const int rows, const int new_row_stride)
{
    int idx = get_global_id(0);
    if (idx >= cols * rows)
        return;
    int row = idx / cols;
    int col = idx % cols;

    float2 smp = {inputX[row * cols + col], inputY[row * cols + col]};
    output[col * new_row_stride + row] = smp;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void InverseTranspose2DPalanarC(__global const float2 *inputX, __global const float2 *inputY, __global float2 *output, const int cols, const int rows, const int sample_per_channel_padd)
{
    int idx = get_global_id(0);
    if (idx >= cols * rows)
        return;
    int row = idx / cols;
    int col = idx % cols;

    float2 smp = {inputX[row * sample_per_channel_padd + col].x, inputY[row * sample_per_channel_padd + col].x};
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
__kernel void convolve1DC(
    __global const float2 *prev,
    __global const float2 *curr,
    __global const float *filter,
    __global float2 *output,
    const int inputSize,
    const int filterSize,
    const int numChannels,
    const int sample_per_channel_padd)
{
    int idx = get_global_id(0); // sample index [0, inputSize]
    if (idx >= inputSize)
        return;

    const int sample_per_channel = inputSize / numChannels;
    const int sample = idx % sample_per_channel;
    const int channel = idx / sample_per_channel;
    const int base = channel * sample_per_channel_padd;
    const int fbase = channel * sample_per_channel_padd;

    float fr;
    float2 smp;
    float2 result = {0.0f, 0.0f};
    for (int f = 0, s = sample + 1 - filterSize; f < filterSize; ++f, ++s)
    {
        fr = filter[fbase + f];
        smp = s < 0 ? prev[base + s + sample_per_channel_padd] : curr[base + s];

        result.x += fr * smp.x;
        result.y += fr * smp.y;
    }
    // output[idx] = result;
    output[channel * sample_per_channel_padd + sample] = result;
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
        smp = s < 0 ? getSample2D(prev, channel, s + sample_per_channel, numChannels) : getSample2D(curr, channel, s, numChannels);

        result.x += fr * smp.x;
        result.y += fr * smp.y;
    }
    output[idx] = result;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void convolve2DFreq(__global float2 *curr, __global const float2 *filter, int size)
{
    int idx = get_global_id(0);
    if (idx >= size)
        return;

    float2 smp = curr[idx];
    float2 fr = filter[idx];

    float2 result = {smp.x * fr.x - smp.y * fr.y, smp.x * fr.y + smp.y * fr.x};

    curr[idx] = result;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void AbsMag(__global const float2 *curr, __global float *out, int size)
{
    int idx = get_global_id(0);
    if (idx >= size)
        return;

    float2 smp = curr[idx];
    out[idx] = smp.x * smp.x + smp.y * smp.y;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__kernel void convolve1DFir(
    __global const float *prev,
    __global const float *curr,
    __global float *output,
    const int sample_per_channel, // 20,000
    const int numChannels)        // 20
{
    const int inputSize = sample_per_channel * numChannels;
    int idx = get_global_id(0); // sample index [0, inputSize]
    if (idx >= inputSize)       // 400,000
        return;

    const float F7 = 1.f / 7.f;
    const int sample = idx / numChannels;
    const int channel = idx % numChannels;
    const int pbase = inputSize + numChannels * 6 + channel;

    float smp;
    float result = 0.0f;
    for (int f = 0, s = sample - 6; f < 7; ++f, ++s)
    {
        smp = s < 0 ? prev[pbase + f * numChannels] : curr[idx - f * numChannels];
        result += smp;
    }
    output[idx] = result * F7;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
