#include "SP.cuh"
#include "cu/Elapse.hpp"
#include "cu/Utils.h"
#include "cu/BMP.h"

NAMESPACE_BEGIN(cu);

//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void Transpose1D(const float *input, float *output, const int cols, const int rows, const int new_row_stride);
__global__ void Transpose1DC(const float *input, float2 *output, const int cols, const int rows, const int new_row_stride, const int reverse);
__global__ void Transpose2D(const float2 *input, float2 *output, const int cols, const int rows, const int new_row_stride, const int offset);
__global__ void Transpose2D_copy_Prev(const float2 *prev, float2 *output, const int cols, const int rows, const int prevRows, const int new_row_stride);
__global__ void InverseTranspose2D(const float2 *input, float2 *output, const int cols, const int rows, const int new_row_stride, const int offset);
__global__ void ExtractChannel(const float2 *input, float2 *output, const int cols, const int rows, const int channel);
__global__ void convolve2DFreq(float2 *curr, const float2 *filter, int size);
__global__ void normalizeSignal(float2 *curr, int size, int N);
__global__ void AbsMag(const float2 *curr, float *out, int size);
__global__ void convolve1DFir(const float *prev, const float *curr, float *output, const int sample_per_channel, const int numChannels);
__global__ void generateHanningWindow(float *window, int length);
__global__ void generateHammingWindow(float *window, int length);
__global__ void applyWindowAndSegmentKernel(const float2 *inputSignal, const float *window, float2 *outputSignal, int signalLength, int windowLength, int hopSize, int numSegments);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
const bool useHost = true;
//-------------------------------------------------------------------------------------------------------------------------------------------------//
SP::SP() : firFilter(0, useHost), currAbs(num_of_samples, useHost), currFir(num_of_samples, useHost),
           filterFFT(num_of_samples_padd, useHost),
           prevT(num_of_samples_padd, useHost), currT(num_of_samples_padd, useHost), signal1(samples_per_channel, useHost), signal1out(num_of_windows * window_size, useHost),
           hanningWindow(window_size, useHost), hammingWindow(window_size, useHost),

           filter(0, useHost), abs_out(num_of_samples, useHost), abs_prev(0, useHost), fir_out(0, useHost),
           fft_out(0, useHost),
           prev(0, useHost), curr(0, useHost), out(num_of_samples, useHost), result(0, useHost)
{
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool SP::ToCSV()
{
    // const auto &workDir = GPU::GetWorkingDirectory() + "../../../data/";
    // std::string outDir = "E:/Dropbox/temp/";

    // std::vector<float> data;
    // std::vector<std::string> names = {"FIR_PREV", "FIR_CURRENT", "FIR_OUT"};
    // for (int i = 0; i < 5; ++i)
    // {
    //     for (const auto &name : names)
    //     {
    //         Utils::FromFile(workDir + name + std::to_string(i) + ".32fc", data);
    //         Utils::ToCSV(outDir + name + std::to_string(i) + ".csv", data.data(), (int)data.size(), num_of_channels);
    //     }
    // }

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool SP::Init()
{
    const auto &workDir = GPU::GetWorkingDirectory() + "../../../data/";
    const float F7 = 1.f / 7.f;
    firFilter.resize(7, F7);

    //    filter.ReadFromFile("../../data/4/filter.bin");
    filter.ReadFromFile(workDir + "FILTER.32fc");
    curr.ReadFromFile(workDir + "Chann_current2.32fc");
    prev.ReadFromFile(workDir + "Chann_prev2.32fc");
    result.ReadFromFile(workDir + "Chann_out2.32fc", false);
    fft_out.ReadFromFile(workDir + "FFT_OUT2.32fc");
    abs_prev.ReadFromFile(workDir + "FIR_PREV2.32fc"); //
    abs_out.ReadFromFile(workDir + "ABS_OUT2.32fc");
    fir_out.ReadFromFile(workDir + "FIR_OUT2.32fc");

    if (filter.size() < num_of_filters || curr.size() != num_of_samples || prev.size() != num_of_samples || result.size() != num_of_samples)
    {
        std::cerr << "Invalid data size" << std::endl;
        return false;
    }

    BMP::SignalReal2BMP(workDir + "FILTER.bmp", filter.hata, (int)filter.size(), 512);

    Elapse el("Filter FFT", 16);
    el.Stamp("Start");

    Transpose1DC<<<DIV(num_of_filters, GRP), GRP>>>(*filter, *filterFFT, num_of_channels, filter_size, samples_per_channel_padd, 1);
    el.Stamp("Transpose Filter");

    for (int i = 0; i < num_of_channels; ++i)
        FFT::Dispatch(true, filterFFT, i * samples_per_channel_padd, samples_per_channel_padd);
    el.Stamp("Filter FFT");

    generateHanningWindow<<<DIV(window_size, GRP), GRP>>>(*hanningWindow, window_size);
    generateHammingWindow<<<DIV(window_size, GRP), GRP>>>(*hammingWindow, window_size);
    el.Stamp("generate windows");

    ExtractChannel<<<DIV(samples_per_channel, GRP), GRP>>>(*curr, *signal1, num_of_channels, samples_per_channel, 0);
    el.Stamp("Extract Channel");

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool SP::Process()
{
    int errors = 0;
    Elapse el("Process", 16);

    Transpose2D<<<DIV(num_of_samples, GRP), GRP>>>(*curr, *currT, num_of_channels, samples_per_channel, samples_per_channel_padd, filter_size);
    Transpose2D_copy_Prev<<<DIV(filter_size * samples_per_channel, GRP), GRP>>>(*prev, *currT, num_of_channels, samples_per_channel, filter_size, samples_per_channel_padd);
    el.Stamp("Transpose Signal");

    for (int i = 0; i < num_of_channels; ++i)
        FFT::Dispatch(true, currT, i * samples_per_channel_padd, samples_per_channel_padd);
    el.Stamp("Signal FFT");

    convolve2DFreq<<<DIV(num_of_samples_padd, GRP), GRP>>>(*currT, *filterFFT, num_of_samples_padd);
    el.Stamp("Convolve");

    for (int i = 0; i < num_of_channels; ++i)
        FFT::Dispatch(false, currT, i * samples_per_channel_padd, samples_per_channel_padd);
    el.Stamp("Signal IFFT");
    normalizeSignal<<<DIV(num_of_samples_padd, GRP), GRP>>>(*currT, num_of_samples_padd, samples_per_channel_padd);
    el.Stamp("Normalize");

    InverseTranspose2D<<<DIV(num_of_samples, GRP), GRP>>>(*currT, *out, samples_per_channel, num_of_channels, samples_per_channel_padd, filter_size);
    el.Stamp("Inverse Transpose");
    FFT::Dispatch(true, out, samples_per_channel);
    el.Stamp("Out FFT");
    AbsMag<<<DIV(num_of_samples, GRP), GRP>>>(*out, *currAbs, num_of_samples);
    el.Stamp("AbsMag");
    convolve1DFir<<<DIV(num_of_samples, GRP), GRP>>>(*abs_prev, *currAbs, *currFir, samples_per_channel, num_of_channels);
    sync();

    errors = Compare(currAbs, abs_out);
    std::cout << "Errors: " << errors << std::endl;

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool SP::STFT()
{
    const auto &workDir = GPU::GetWorkingDirectory() + "../../../data/";
    Elapse el("STFT", 4);

    dim3 blocks(DIV(window_size, GRP), num_of_windows);
    applyWindowAndSegmentKernel<<<blocks, GRP>>>(*signal1, *hammingWindow, *signal1out, samples_per_channel, window_size, hop_size, num_of_windows);
    el.Stamp("Apply Window");
    FFT::dispatch(true, *signal1out, *signal1out, window_size, num_of_windows, 0);
    el.Stamp("STFT");
    sync();
    BMP::STFTComplex2BMP(workDir + "stft.bmp", signal1out.hata, window_size, num_of_windows);

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void Transpose1D(const float *input, float *output, const int cols, const int rows, const int new_row_stride)
{
    int idx = getI();
    if (idx >= cols * rows)
        return;
    int row = idx / cols;
    int col = idx % cols;

    output[col * new_row_stride + row] = input[row * cols + col];
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void Transpose1DC(const float *input, float2 *output, const int cols, const int rows, const int new_row_stride, const int reverse)
{
    int idx = getI();
    if (idx >= cols * rows)
        return;
    int row = idx / cols;
    int col = idx % cols;
    int nrow = 0 == reverse ? row : rows - 1 - row;
    float2 v = {input[row * cols + col], 0.0f};
    output[col * new_row_stride + nrow] = v;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void Transpose2D(const float2 *input, float2 *output, const int cols, const int rows, const int new_row_stride, const int offset)
{
    // cols = 20, rows = 20,000, new_row = 21,600
    int idx = getI();
    if (idx >= cols * rows) // 400,000
        return;
    int row = idx / cols; // [0,20,000)
    int col = idx % cols; // [0,20)

    output[col * new_row_stride + row + offset] = input[row * cols + col];
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void Transpose2D_copy_Prev(const float2 *prev, float2 *output, const int cols, const int rows, const int prevRows, const int new_row_stride)
{
    // cols = 20, rows = 20,000, prevRows = 500, new_row = 21,600
    int idx = getI();
    if (idx >= cols * prevRows) // 10,000
        return;
    int row = idx / cols;                // [0,500)
    int col = idx % cols;                // [0,20)
    int prevRow = rows - prevRows + row; // [19500,20000)

    output[col * new_row_stride + row] = prev[prevRow * cols + col];
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void InverseTranspose2D(const float2 *input, float2 *output, const int cols, const int rows, const int new_row_stride, const int offset)
{
    // cols = 20, rows = 20,000, new_row = 21,600
    int idx = getI();
    if (idx >= cols * rows) // 400,000
        return;
    int row = idx / cols; // [0,20,000)
    int col = idx % cols; // [0,20)

    output[col * rows + row] = input[row * new_row_stride + col + offset];
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void ExtractChannel(const float2 *input, float2 *output, const int cols, const int rows, const int channel)
{
    int idx = getI();
    if (idx >= rows)
        return;

    output[idx] = input[idx * cols + channel];
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void convolve2DFreq(float2 *curr, const float2 *filter, int size)
{
    int idx = getI();
    if (idx >= size)
        return;

    float2 smp = curr[idx];
    float2 fr = filter[idx];

    float2 result = {smp.x * fr.x - smp.y * fr.y, smp.x * fr.y + smp.y * fr.x};

    curr[idx] = result;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void normalizeSignal(float2 *curr, int size, int N)
{
    int idx = getI();
    if (idx >= size)
        return;

    float2 smp = curr[idx];
    smp.x /= N;
    smp.y /= N;
    curr[idx] = smp;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void AbsMag(const float2 *curr, float *out, int size)
{
    int idx = getI();
    if (idx >= size)
        return;

    float2 smp = curr[idx];
    out[idx] = smp.x * smp.x + smp.y * smp.y;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void convolve1DFir(
    const float *prev,
    const float *curr,
    float *output,
    const int sample_per_channel, // 20,000
    const int numChannels)        // 20
{
    const int inputSize = sample_per_channel * numChannels;
    int idx = getI();     // sample index [0, inputSize]
    if (idx >= inputSize) // 400,000
        return;

    const float F7 = 1.f / 7.f;
    const int sample = idx / numChannels;
    // const int channel = idx % numChannels;
    if (sample < 7)
        return;

    float smp;
    float result = 0.0f;
    for (int f = 0; f < 7; ++f)
    {
        //        smp = s < 0 ? prev[base + s + sample_per_channel] : curr[base + s];
        //        smp = s < 0 ? getSample1D(prev, channel, s + sample_per_channel, numChannels) : getSample1D(curr, channel, s, numChannels);
        smp = curr[idx - f * numChannels];
        result += smp;
    }
    output[idx] = result * F7;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void generateHanningWindow(float *window, int length)
{
    int idx = getI(); // [0, length)
    if (idx >= length)
        return;

    window[idx] = 0.5 * (1 - cos(2 * M_PI * idx / (length - 1)));
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void generateHammingWindow(float *window, int length)
{
    int idx = getI(); // [0, length)
    if (idx >= length)
        return;

    window[idx] = 0.54 - 0.46 * cos(2 * M_PI * idx / (length - 1));
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void applyWindowAndSegmentKernel(
    const float2 *inputSignal, // input signal      20,000
    const float *window,       // window function   1024
    float2 *outputSignal,      // output signal     38912 = 38 * 1024
    int signalLength,          // 20,000
    int windowLength,          // 1024
    int hopSize,               // 512
    int numSegments)           // 38
{
    int idx = getI(); // [0, 1024)
    int jdx = getJ(); // [0, 38)
    if (idx >= windowLength || jdx >= numSegments || jdx * hopSize + idx >= signalLength)
        return;

    float winValue = window[idx];
    int ibase = jdx * hopSize + idx;
    int obase = jdx * windowLength + idx;

    outputSignal[obase].x = inputSignal[ibase].x * winValue;
    outputSignal[obase].y = inputSignal[ibase].y * winValue;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(cu);