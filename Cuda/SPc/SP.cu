#include "SP.cuh"
#include "cu/Elapse.hpp"
#include "cu/Utils.h"
#include "cu/BMP.cuh"

NAMESPACE_BEGIN(cu);

//-------------------------------------------------------------------------------------------------------------------------------------------------//
const int FRESL = 1000000;
const bool useHost = false;
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void FillDFTWeights(float2 *weights, int N);
__global__ void MultiDFT(float2 *output, const float2 *data, const float2 *weights, int size, int N);
__global__ void MultiDFTIL(float2 *output, const float2 *data, const float2 *weights, const int cols, const int rows, const int new_row_stride, const int offset);
__global__ void Transpose1D(const float *input, float *output, const int cols, const int rows, const int new_row_stride);
__global__ void Transpose1DC(const float *input, float2 *output, const int cols, const int rows, const int new_row_stride, const int reverse);
__global__ void Transpose2D(const float2 *input, float2 *output, const int cols, const int rows, const int new_row_stride, const int offset);
__global__ void Transpose2D_copy_Prev(const float2 *prev, float2 *output, const int cols, const int rows, const int prevRows, const int new_row_stride);
__global__ void InverseTranspose2D(const float2 *input, float2 *output, const int cols, const int rows, const int new_row_stride, const int offset);
__global__ void DeChirp(const float2 *input, float2 *output, size_t size, double MFS);
__global__ void CreateSignal(float2 *output, const int size);
__global__ void convolve2DFreq(float2 *curr, const float2 *filter, int size);
__global__ void normalizeSignal(float2 *curr, int size, int N);
__global__ void AbsMag(const float2 *curr, float *out, int size);
__global__ void convolve1DDown(const float2 *curr, const float *filter, float2 *output, const int size, const int downSize, const int filterSize, const int downSample, const int offset);
__global__ void convolve1DFir(const float *prev, const float *curr, float *output, const int sample_per_channel, const int numChannels);
__global__ void convolve1DFirIL(const float *prev, const float *curr, float *output, const int sample_per_channel, const int numChannels);
__global__ void generateOnesWindow(float *window, int length);
__global__ void generateHanningWindow(float *window, int length);
__global__ void generateHammingWindow(float *window, int length);
__global__ void applyOverlapSignal(const float2 *inputSignal, float2 *outputSignal, int signalSize, int windowSize, int hopSize, int numOfWindows);
__global__ void applyWindowAndSegment(const float2 *inputSignal, const float *window, float2 *outputSignal, int signalLength, int windowLength, int hopSize, int numSegments);
__global__ void fftShift(float2 *buffer, int windowSize, int numOfWindows);
__global__ void MinMaxAtomic(const float2 *input, int *output, int size);
__global__ void MinMaxAtomicBlock(const float2 *input, int *output, int size);
__global__ void MinMaxAtomicBlockShfl(const float2 *input, int *output, int size);
__global__ void MinMaxAtomicBlockChunk(const float2 *input, int *output, int chunk, int size);
__global__ void MinMaxReduction(const float2 *input, int *output, int size);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
SP::SP() : globals(2048, useHost),
           firFilter(0, useHost), currAbs(num_of_samples, useHost), currFir(num_of_samples, useHost),
           filterFFT(num_of_samples_padd, useHost), weightsDFT(num_of_channels * num_of_channels, useHost),
           prevT(num_of_samples_padd, useHost), currT(num_of_samples_padd, useHost), signal1(test_size, useHost), signal1out(num_of_windows * window_size, useHost), signal1outT(num_of_windows * window_size, useHost),
           hanningWindow(window_size, useHost), hammingWindow(window_size, useHost), onesWindow(window_size, useHost),
           fir2(0, useHost), fir4(0, useHost), fir8(0, useHost),

           dechirp(0, useHost), signalDown(0, useHost), overlapSignal(0, useHost),

           filter(0, useHost), abs_out(num_of_samples, useHost), abs_prev(0, useHost), fir_out(0, useHost),
           fft_out(0, useHost),
           prev(0, useHost), curr(0, useHost), out(num_of_samples, useHost), tout(num_of_samples, useHost), result(0, useHost)
{
#ifdef _WIN32
    workDir = GPU::GetWorkingDirectory() + "../../../data/";
#else
    workDir = "/tmp/data/";
#endif
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
// #define NON_TRANSPOSE
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool SP::Init()
{
    BMP::Get();

    const float F7 = 1.f / 7.f;
    firFilter.resize(7, F7);

    //    filter.ReadFromFile("../../data/4/filter.bin");
    filter.ReadFromFile(workDir + "FILTER.32fc");
    curr.ReadFromFile(workDir + "Chann_current2.32fc");
    curr.ReadFromFile(workDir + "Chann_current2.32fc");
    prev.ReadFromFile(workDir + "Chann_prev2.32fc");
    result.ReadFromFile(workDir + "Chann_out2.32fc", false);
    fft_out.ReadFromFile(workDir + "FFT_OUT2.32fc");
    abs_prev.ReadFromFile(workDir + "FIR_PREV2.32fc"); //
    abs_out.ReadFromFile(workDir + "ABS_OUT2.32fc");
    fir_out.ReadFromFile(workDir + "FIR_OUT2.32fc");
    chann1.ReadFromFile(workDir + "Bluritsamples/Test0_input.32fc");
    fir2.ReadFromFile(workDir + "Bluritsamples/fir2.32f");
    fir4.ReadFromFile(workDir + "Bluritsamples/fir4.32f");
    fir8.ReadFromFile(workDir + "Bluritsamples/fir8.32f");

    if (filter.size() < num_of_filters || curr.size() != num_of_samples || prev.size() != num_of_samples || result.size() != num_of_samples)
    {
        std::cerr << "Invalid data size" << std::endl;
        return false;
    }

    // BMP::SignalReal2BMP(workDir + "FILTER.bmp", filter, (int)filter.size(), 512, 1);

    Elapse el("Filter FFT", 16);

    Transpose1DC<<<DIV(num_of_filters, GRP), GRP>>>(*filter, *filterFFT, num_of_channels, filter_size, samples_per_channel_padd, 1);
    el.Stamp("Transpose Filter");

    FFT::Dispatch(true, filterFFT, num_of_channels);
    el.Stamp("Filter FFT");
    FillDFTWeights<<<DIV(num_of_channels * num_of_channels, GRP), GRP>>>(*weightsDFT, num_of_channels);
    el.Stamp("Weights DFT");

    generateOnesWindow<<<DIV(window_size, GRP), GRP>>>(*onesWindow, window_size);
    generateHanningWindow<<<DIV(window_size, GRP), GRP>>>(*hanningWindow, window_size);
    generateHammingWindow<<<DIV(window_size, GRP), GRP>>>(*hammingWindow, window_size);
    el.Stamp("generate windows");

    chann1Padd.resize(FFT::NextPow2((int)chann1.size()));
    Copy(*chann1Padd, *chann1, chann1.size());
    chann1FFT.resize(chann1Padd.size());

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool SP::XYZProcess()
{
    int errors = 0, k = 0, L = 30;
    Elapse el("XYZProcess", 16);

    for (k = 0; k < 1000; ++k)
    {
        el.Loop("test", true, k < L);
        Transpose2D<<<DIV(num_of_samples, GRP), GRP>>>(*curr, *currT, num_of_channels, samples_per_channel, samples_per_channel_padd, filter_size);
        Transpose2D_copy_Prev<<<DIV(filter_size * samples_per_channel, GRP), GRP>>>(*prev, *currT, num_of_channels, samples_per_channel, filter_size, samples_per_channel_padd);
        el.Stamp("Transpose Signal", k < L);

        FFT::Dispatch(true, currT, num_of_channels);
        convolve2DFreq<<<DIV(num_of_samples_padd, GRP), GRP>>>(*currT, *filterFFT, num_of_samples_padd);
        FFT::Dispatch(false, currT, num_of_channels);
        normalizeSignal<<<DIV(num_of_samples_padd, GRP), GRP>>>(*currT, num_of_samples_padd, samples_per_channel_padd);
        el.Stamp("Convolve", k < L);

#ifdef NON_TRANSPOSE
        MultiDFTIL<<<DIV(num_of_samples, GRP), GRP>>>(*out, *currT, *weightsDFT, samples_per_channel, num_of_channels, samples_per_channel_padd, filter_size);
        el.Stamp("Multi DFT", k < L);
        AbsMag<<<DIV(num_of_samples, GRP), GRP>>>(*out, *currAbs, num_of_samples);
        el.Stamp("AbsMag", k < L);
        convolve1DFir<<<DIV(num_of_samples, GRP), GRP>>>(*abs_prev, *currAbs, *currFir, samples_per_channel, num_of_channels);
        el.Stamp("Fir", k < L);
#else
        InverseTranspose2D<<<DIV(num_of_samples, GRP), GRP>>>(*currT, *out, samples_per_channel, num_of_channels, samples_per_channel_padd, filter_size);
        el.Stamp("Inverse Transpose", k < L);
        FFT::Dispatch(true, out, samples_per_channel);
        el.Stamp("Out FFT", k < L);
        AbsMag<<<DIV(num_of_samples, GRP), GRP>>>(*out, *currAbs, num_of_samples);
        el.Stamp("AbsMag", k < L);
        convolve1DFirIL<<<DIV(num_of_samples, GRP), GRP>>>(*abs_prev, *currAbs, *currFir, samples_per_channel, num_of_channels);
        el.Stamp("FirIL", k < L);
#endif
        el.Loop("test", false, k < L);
    }
#ifdef NON_TRANSPOSE
    Transpose1D<<<DIV(num_of_samples, GRP), GRP>>>(*currAbs, *currFir, samples_per_channel, num_of_channels, num_of_channels);
    currAbs.swap(currFir);
#endif
    sync();

    errors = Compare(currAbs, abs_out);
    std::cout << "Errors: " << errors << std::endl;

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__host__ __device__ float2 ChirpK(int k, double InvSF)
{
    float2 f;
    double t = double(k) * InvSF * (k + 1) * 0.5;
    //    t *= (k+1)*0.5; // t = (tk - t0)*n/2;

    double p = -2.0 * M_PI * t;
    f = {(float)cos(p), (float)sin(p)};
    return f;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool SP::ZYXLoadFile(int channelId, int &downSample, int &signalSize, int &windowSize, int &hopSize, int &numOfWindows, int &SF, bool saveImage)
{
    const int CHANNEL_NUM = channelId;
    SF = ((CHANNEL_NUM - 1) / 4) + 7;
    const int SYMBOL = 1 << SF;

    downSample = 1 << (3 - (CHANNEL_NUM - 1) % 4);

    std::string channel = std::to_string(CHANNEL_NUM);

    bool rv = true;
    rv = rv && decimate1.ReadFromFile(workDir + "Bluritsamples/Test0_decimated_num_" + channel + ".32fc");
    rv = rv && chirp1.ReadFromFile(workDir + "Bluritsamples/chirp" + std::to_string(downSample) + ".32fc");
    rv = rv && dechirp1.ReadFromFile(workDir + "Bluritsamples/Test0_dechirp_num_" + channel + ".32fc");
    if (false == rv)
        return false;

    const float overlap = 0.75f;

    windowSize = SYMBOL;
    signalSize = (int)dechirp1.size();
    int signalPaddSize = DIV(signalSize, windowSize) * windowSize;
    int overlapSize = (int)(windowSize * overlap);
    hopSize = windowSize - overlapSize;
    numOfWindows = signalPaddSize / hopSize;

    rv = rv && stft1.ReadFromFile(workDir + "Bluritsamples/Test0_" + std::to_string(SYMBOL) + "x" + std::to_string(numOfWindows) + "_stft_blocks_num_" + channel + ".32fc");
    if (false == rv)
        return false;

    if (false == saveImage)
        return true;

    BMP::SignalComplex2BMP(workDir + "x_inp" + channel + ".bmp", chann1, 20000, 128, eType::Real);
    BMP::SignalComplex2BMP(workDir + "x_dec" + channel + ".bmp", decimate1, 20000, 128, eType::Real);
    BMP::SignalComplex2BMP(workDir + "x_crp" + channel + ".bmp", chirp1, 20000, 128, eType::Real);
    BMP::SignalComplex2BMP(workDir + "x_dch" + channel + ".bmp", dechirp1, 20000, 128, eType::Real);
    BMP::STFTComplex2BMP(workDir + "x_stft" + channel + ".bmp", stft1, windowSize, numOfWindows, numOfWindows, windowSize, eType::Magnitute);

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool SP::SingleZYXProcess(int channelId)
{
    auto channel = std::to_string(channelId);
    int downSample, windowSize, signalSize, hopSize, numOfWindows, SF;
    if (false == ZYXLoadFile(channelId, downSample, signalSize, windowSize, hopSize, numOfWindows, SF, false))
    {
        std::cout << "Failed to load files\n";
        return false;
    }
    double MFS = 1.0 / (1 << SF);
    int k = 0, L = 30, LOOPS = 100;
    Elapse el("Single ZYX Channel", 16);

    const int downSize = DIV((int)chann1.size(), downSample);
    signalDown.resize(downSize);
    dechirp.resize(downSize);
    overlapSignal.resize(numOfWindows * windowSize);
    int errors = 0;

    for (k = 0; k < LOOPS; ++k)
    {
        el.Loop("SingleZYXProcess", true, k < L);
        gbuffer<float> *pFir = nullptr;
        switch (downSample)
        {
        case 2:
            pFir = &fir2;
            break;
        case 4:
            pFir = &fir4;
            break;
        case 8:
            pFir = &fir8;
            break;
        }
        if (pFir)
        {
            auto &fir = *pFir;
            const int filterSize = (int)fir.size();
            const int offset = (filterSize / 2) / downSample;

            convolve1DDown<<<DIV(downSize, GRP), GRP>>>(*chann1, *fir, *signalDown, (int)chann1.size(), downSize, filterSize, downSample, offset);
        }
        else
        {
            Copy(*signalDown, *chann1, downSize);
        }
        el.Stamp("Decimate", k < L);

        DeChirp<<<DIV((int)decimate1.size(), GRP), GRP>>>(*signalDown, *dechirp, dechirp.size(), MFS);
        el.Stamp("DeChirp", k < L);

        dim3 grid(DIV(windowSize, GRP), numOfWindows);
        applyOverlapSignal<<<grid, GRP>>>(*dechirp, *overlapSignal, signalSize, windowSize, hopSize, numOfWindows);
        el.Stamp("Apply Window");

        FFT::Dispatch(true, overlapSignal, numOfWindows);
        fftShift<<<grid, GRP>>>(*overlapSignal, windowSize, numOfWindows);
        el.Stamp("STFT");

        el.Loop("SingleZYXProcess", false, k < L);
    }
    sync();
    errors = Compare(decimate1, signalDown, decimate1.size());
    std::cout << "Decimate Chann [" + channel + "] Errors: " << errors << std::endl;

    errors = Compare(dechirp1, dechirp, dechirp1.size());
    std::cout << "DeChirp [" + channel + "] Errors: " << errors << std::endl;

    errors = Compare(stft1, overlapSignal, stft1.size());
    std::cout << "STFT Chann [" + channel + "] Errors: " << errors << std::endl;
    
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool SP::MinMax()
{
    int k = 0, L = 30, LOOPS = 1000;
    Elapse el("MinMax", 16);

    float cpuVal[2] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::min()};
    for (const auto &c : curr.cpu())
    {
        cpuVal[0] = std::min(cpuVal[0], c.x);
        cpuVal[1] = std::max(cpuVal[1], c.x);
    }

    globals[0] = FRESL;
    globals[1] = -FRESL;
    globals.RefreshUp(2);
    for (k = 0; k < LOOPS; ++k)
    {
        el.Loop("atomics", true, k < L);

        MinMaxAtomic<<<DIV(num_of_samples, GRP), GRP>>>(*curr, *globals, num_of_samples);

        el.Loop("atomics", false, k < L);
    }
    globals[0] = FRESL;
    globals[1] = -FRESL;
    globals.RefreshUp(2);
    for (k = 0; k < LOOPS; ++k)
    {
        el.Loop("atomicsBlock", true, k < L);

        MinMaxAtomicBlock<<<DIV(num_of_samples, GRP), GRP>>>(*curr, *globals, num_of_samples);

        el.Loop("atomicsBlock", false, k < L);
    }

    globals[0] = FRESL;
    globals[1] = -FRESL;
    globals.RefreshUp(2);
    for (k = 0; k < LOOPS; ++k)
    {
        el.Loop("atomicsBlockShfl", true, k < L);

        MinMaxAtomicBlockShfl<<<DIV(num_of_samples, GRP), GRP>>>(*curr, *globals, num_of_samples);
        // sync();
        // globals.RefreshDown(2);
        el.Loop("atomicsBlockShfl", false, k < L);
    }

    globals[0] = FRESL;
    globals[1] = -FRESL;
    globals.RefreshUp(2);
    for (k = 0; k < LOOPS; ++k)
    {
        el.Loop("reduction", true, k < L);

        MinMaxReduction<<<DIV(num_of_samples, GRP), GRP>>>(*curr, *globals, num_of_samples);
        // sync();
        // globals.RefreshDown(2);
        el.Loop("reduction", false, k < L);
    }

    globals[0] = FRESL;
    globals[1] = -FRESL;
    globals.RefreshUp(2);
    int chunk = 8;
    for (k = 0; k < LOOPS; ++k)
    {
        el.Loop("atomicsBlockChunk", true, k < L);

        MinMaxAtomicBlockChunk<<<DIV(num_of_samples, GRP * chunk), GRP>>>(*curr, *globals, chunk, num_of_samples);
        // sync();
        // globals.RefreshDown(2);
        el.Loop("atomicsBlockChunk", false, k < L);
    }
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void MinMaxAtomic(const float2 *input, int *output, int size)
{
    int idx = getI();
    if (idx >= size)
        return;

    int smp = input[idx].x * FRESL;
    atomicMin(&output[0], smp);
    atomicMax(&output[1], smp);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void MinMaxAtomicI(const int *input, int *output, int size)
{
    int idx = getI();
    if (idx >= size)
        return;

    int smp = input[idx];
    atomicMin(&output[0], smp);
    atomicMax(&output[1], smp);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void MinMaxAtomicBlock(const float2 *input, int *output, int size)
{
    int idx = getI();
    if (idx >= size)
        return;
    __shared__ int gl[2];
    if (threadIdx.x == 0)
    {
        gl[0] = FRESL;
        gl[1] = -FRESL;
    }
    __syncthreads();

    int smp = input[idx].x * FRESL;

    atomicMin_block(&gl[0], smp);
    atomicMax_block(&gl[1], smp);
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicMin(&output[0], gl[0]);
        atomicMax(&output[1], gl[1]);
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__device__ void warpReduceMinMax(int &minVal, int &maxVal)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        int shfl_min = __shfl_down_sync(0xffffffff, minVal, offset);
        int shfl_max = __shfl_down_sync(0xffffffff, maxVal, offset);
        minVal = min(minVal, shfl_min);
        maxVal = max(maxVal, shfl_max);
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void MinMaxAtomicBlockShfl(const float2 *input, int *output, int size)
{
    int idx = getI();
    if (idx >= size)
        return;

    const int wrapCount = blockDim.x / warpSize; // 8
    int laneId = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    __shared__ int gl[2];
    __shared__ int vals[8][2];
    if (threadIdx.x == 0)
    {
        gl[0] = FRESL;
        gl[1] = -FRESL;
    }
    __syncthreads();

    int smp = input[idx].x * FRESL;
    int minVal = smp, maxVal = smp;
    warpReduceMinMax(minVal, maxVal);
    if (laneId == 0)
    {
        vals[warpId][0] = minVal;
        vals[warpId][1] = maxVal;
    }
    __syncthreads();
    if (threadIdx.x < wrapCount)
    {
        minVal = vals[threadIdx.x][0];
        maxVal = vals[threadIdx.x][1];
    }
    else
    {
        minVal = FRESL;
        maxVal = -FRESL;
    }
    if (warpId == 0)
        warpReduceMinMax(minVal, maxVal);
    if (laneId == 0)
    {
        atomicMin_block(&gl[0], minVal);
        atomicMax_block(&gl[1], maxVal);
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicMin(&output[0], gl[0]);
        atomicMax(&output[1], gl[1]);
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void MinMaxAtomicBlockChunk(const float2 *input, int *output, int chunk, int size)
{
    int idx = getI() * chunk;
    // if (idx >= size)
    //     return;
    __shared__ int gl[2];
    if (threadIdx.x == 0)
    {
        gl[0] = FRESL;
        gl[1] = -FRESL;
    }
    __syncthreads();

    int smp;
    int localMin = FRESL;
    int localMax = -FRESL;

    for (int i = 0; i < chunk && (idx + i) < size; ++i)
    {
        smp = input[idx + i].x * FRESL;
        localMin = min(localMin, smp);
        localMax = max(localMax, smp);
    }

    atomicMin_block(&gl[0], localMin);
    atomicMax_block(&gl[1], localMax);
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicMin(&output[0], gl[0]);
        atomicMax(&output[1], gl[1]);
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void MinMaxReduction(const float2 *input, int *output, int size)
{
    __shared__ int sdata[256][2];

    int idx = getI();
    int tid = threadIdx.x;

    if (idx < size)
    {
        int smp = input[idx].x * FRESL;

        sdata[tid][0] = smp;
        sdata[tid][1] = smp;
    }
    else
    {
        sdata[tid][0] = FRESL;
        sdata[tid][1] = -FRESL;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && (idx + s) < size)
        {
            sdata[tid][0] = min(sdata[tid][0], sdata[tid + s][0]);
            sdata[tid][1] = max(sdata[tid][1], sdata[tid + s][1]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicMin(&output[0], sdata[0][0]);
        atomicMax(&output[1], sdata[0][1]);
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void FillDFTWeights(float2 *weights, int N)
{
    int idx = getI();
    if (idx >= N * N)
        return;

    int i = idx / N;
    int j = idx % N;
    const float invN = 1.f / float(N);

    float angle = -2.f * M_PI * i * j * invN;
    weights[idx] = {cos(angle), sin(angle)};
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void MultiDFT(float2 *output, const float2 *data, const float2 *weights, int size, int N)
{
    // size = 400,000, N = 20
    int idx = getI(); // [0, 400,000)
    if (idx >= size)  // 400,000
        return;

    int row = idx / N; // [0,20,000)
    int col = idx % N; // [0,20)

    float2 smp, w;
    float2 res = {0.f, 0.f};
    for (int k = 0; k < N; ++k)
    {
        w = weights[col * N + k];
        smp = data[row * N + k];
        res.x += smp.x * w.x - smp.y * w.y;
        res.y += smp.x * w.y + smp.y * w.x;
    }
    output[idx] = res;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void MultiDFTIL(float2 *output, const float2 *data, const float2 *weights, const int cols, const int rows, const int new_row_stride, const int offset)
{
    // cols = 20,000, rows = 20, new_row = 32,768
    // data ->              [32,768]x[20]
    // output ->            [20,000]x[20]
    int idx = getI();       // [0, 400,000)
    if (idx >= cols * rows) // 400,000
        return;

    int row = idx / cols; // [0,20)
    int col = idx % cols; // [0,20,000)

    float2 smp, w;
    float2 res = {0.f, 0.f};
    for (int k = 0; k < rows; ++k)
    {
        w = weights[row * rows + k];
        smp = data[k * new_row_stride + col + offset];
        res.x += smp.x * w.x - smp.y * w.y;
        res.y += smp.x * w.y + smp.y * w.x;
    }
    output[idx] = res;
    //    output[col * rows + row] = res;
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
    // cols = 20,000, rows = 20, new_row = 32,768
    int idx = getI();       // [0, 400,000)
    if (idx >= cols * rows) // 400,000
        return;
    int row = idx / cols; // [0,20)
    int col = idx % cols; // [0,20,000)

    output[col * rows + row] = input[row * new_row_stride + col + offset];
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void CreateSignal(float2 *output, const int size)
{
    int idx = getI();
    if (idx >= size)
        return;

    const float F0 = 1.f, F1 = 500.f;
    const float fs = float(size);
    float t = (float)idx / fs;
    float k = F1 - F0;
    float freq = F0 * t + 0.5 * k * t;
    //    freq = 100.f;
    float phase = 2.f * M_PI * freq * t;
    float x = cos(phase);
    float y = sin(phase);
    float2 yval = {x, y};
    output[idx] = yval;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void DeChirp(const float2 *input, float2 *output, size_t size, double MFS)
{
    int idx = getI();
    if (idx >= size)
        return;

    float2 chirp_sample = ChirpK(idx, MFS);
    float2 sample = input[idx];

    float x1 = sample.x, y1 = sample.y;
    float x2 = chirp_sample.x, y2 = chirp_sample.y;

    float2 result = {x1 * x2 - y1 * y2, x1 * y2 + x2 * y1};

    output[idx] = result;
}
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
__global__ void convolve1DDown(const float2 *curr, const float *filter, float2 *output, const int size, const int downSize, const int filterSize, const int downSample, const int offset)
{
    int idx = getI();
    if (idx >= downSize)
        return;

    const int I = (idx + offset) * downSample;

    float2 res = {0.f, 0.f};
    for (int f = 0, s = I + 1 - filterSize; f < filterSize; ++f, ++s)
    {
        if (s >= 0 && s < size)
            res += curr[s] * filter[f];
    }
    output[idx] = res;
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
        smp = curr[idx - f];
        result += smp;
    }
    output[idx] = result * F7;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void convolve1DFirIL(
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
__global__ void generateOnesWindow(float *window, int length)
{
    int idx = getI(); // [0, length)
    if (idx >= length)
        return;

    window[idx] = 1.f;
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
__global__ void applyOverlapSignal(
    const float2 *inputSignal, // unpadded signal
    float2 *outputSignal,      // output size = windowSize * numOfWindows
    int signalSize,            // unpadded signal size
    int windowSize,
    int hopSize,
    int numOfWindows)
{
    const int idx = getI(); // [0, windowSize)
    const int jdx = getJ(); // [0, numOfWindows)
    if (idx >= windowSize || jdx >= numOfWindows)
        return;

    const int overlapSize = windowSize - hopSize;
    const int ibase = jdx * hopSize + idx - overlapSize;
    const int obase = jdx * windowSize + idx;

    if (0 <= ibase && ibase < signalSize)
    {
        outputSignal[obase].x = inputSignal[ibase].x;
        outputSignal[obase].y = inputSignal[ibase].y;
    }
    else
    {
        outputSignal[obase].x = 0.f;
        outputSignal[obase].y = 0.f;
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void applyWindowAndSegment(
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
__global__ void fftShift(float2 *buffer, int windowSize, int numOfWindows)
{
    int idx = getI(); // [0, windowSize/2)
    int jdx = getJ(); // [0, numOfWindows)
    if (idx >= windowSize / 2 || jdx >= numOfWindows)
        return;

    int mid = windowSize / 2;
    int base = jdx * windowSize;
    int i = base + idx;
    int j = base + idx + mid;

    float2 t = buffer[i];
    buffer[i] = buffer[j];
    buffer[j] = t;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(cu);
