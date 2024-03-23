#include "SP.cuh"
#include "cu/Elapse.hpp"
#include "cu/Utils.h"
#include "cu/BMP.cuh"

NAMESPACE_BEGIN(cu);

//-------------------------------------------------------------------------------------------------------------------------------------------------//
const int CHUNK_SIZE = 1 << 18;
const bool useHost = false;
const float OVERLAP = 0.75f;
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__host__ __device__ float2 CW(double t);
__global__ void FillDFTWeights(float2 *weights, int N);
__global__ void MultiDFT(float2 *output, const float2 *data, const float2 *weights, int size, int N);
__global__ void MultiDFTIL(float2 *output, const float2 *data, const float2 *weights, const int cols, const int rows, const int new_row_stride, const int offset);
__global__ void Transpose1D(const float *input, float *output, const int cols, const int rows, const int new_row_stride);
__global__ void Transpose1DC(const float *input, float2 *output, const int cols, const int rows, const int new_row_stride, const int reverse);
__global__ void Transpose2D(const float2 *input, float2 *output, const int cols, const int rows, const int new_row_stride, const int offset);
__global__ void Transpose2D_copy_Prev(const float2 *prev, float2 *output, const int cols, const int rows, const int prevRows, const int new_row_stride);
__global__ void InverseTranspose2D(const float2 *input, float2 *output, const int cols, const int rows, const int new_row_stride, const int offset);
__global__ void DeChirp(const float2 *input, float2 *output, size_t size, double MFS, int chirpOffset, int chirpSize);
__global__ void convolve2DFreq(float2 *curr, const float2 *filter, int size);
__global__ void normalizeSignal(float2 *curr, int size, int N);
__global__ void AbsMag(const float2 *curr, float *out, int size);
__global__ void convolve1DDown(const float2 *curr, const float *filter, float2 *output, const int size, const int downSize, const int filterSize, const int downSample, const int offset);
__global__ void convolve1DDownPrev(const float2 *prev, const float2 *curr, const float *filter, float2 *output, const int size, const int downSize, const int filterSize, const int downSample, const int offset);
__global__ void convolve1DFir(const float *prev, const float *curr, float *output, const int sample_per_channel, const int numChannels);
__global__ void convolve1DFirIL(const float *prev, const float *curr, float *output, const int sample_per_channel, const int numChannels);
__global__ void generateOnesWindow(float *window, int length);
__global__ void generateHanningWindow(float *window, int length);
__global__ void generateHammingWindow(float *window, int length);
__global__ void applyOverlapSignal(const float2 *inputSignal, float2 *outputSignal, int signalSize, int windowSize, int hopSize, int m_numOfWindows);
__global__ void applyOverlapSignalPrev(const float2 *prev, const float2 *curr, float2 *outputSignal, int signalSize, int windowSize, int hopSize, int m_numOfWindows);
__global__ void applyWindowAndSegment(const float2 *inputSignal, const float *window, float2 *outputSignal, int signalLength, int windowLength, int hopSize, int numSegments);
__global__ void freqShift(float2 *buffer, int size, double invFS);
__global__ void fftShift(float2 *buffer, int windowSize, int m_numOfWindows);
__global__ void calcStatsPerWindow(float4 *output, const float2 *input, int windowSize, int m_numOfWindows);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
SP::SP() : globals(2048, useHost),
           firFilter(0, useHost), currAbs(num_of_samples, useHost), currFir(num_of_samples, useHost),
           filterFFT(num_of_samples_padd, useHost), weightsDFT(num_of_channels * num_of_channels, useHost),
           prevT(num_of_samples_padd, useHost), currT(num_of_samples_padd, useHost),
           fir2(0, useHost), fir4(0, useHost), fir8(0, useHost),

           signalDown(0, useHost), overlapSignal(0, useHost),

           filter(0, useHost), abs_out(num_of_samples, useHost), abs_prev(0, useHost), fir_out(0, useHost),
           fft_out(0, useHost),
           prev(0, useHost), curr(0, useHost), out(num_of_samples, useHost), tout(num_of_samples, useHost), result(0, useHost)
{
#ifdef _WIN32
    workDir = GPU::GetWorkingDirectory() + "../../../data/";
#else
    workDir = "/tmp/data/";
#endif

    supDir = workDir + "superman/";
    spdDir = workDir + "spiderman/";
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
// #define NON_TRANSPOSE
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool SP::SupermanInit()
{
    BMP::Get();

    const float F7 = 1.f / 7.f;
    firFilter.resize(7, F7);

    filter.ReadFromFile(supDir + "FILTER.32fc");
    curr.ReadFromFile(supDir + "Chann_current2.32fc");
    curr.ReadFromFile(supDir + "Chann_current2.32fc");
    prev.ReadFromFile(supDir + "Chann_prev2.32fc");
    result.ReadFromFile(supDir + "Chann_out2.32fc", false);
    fft_out.ReadFromFile(supDir + "FFT_OUT2.32fc");
    abs_prev.ReadFromFile(supDir + "FIR_PREV2.32fc"); //
    abs_out.ReadFromFile(supDir + "ABS_OUT2.32fc");
    fir_out.ReadFromFile(supDir + "FIR_OUT2.32fc");

    if (filter.size() < num_of_filters || curr.size() != num_of_samples || prev.size() != num_of_samples || result.size() != num_of_samples)
    {
        std::cerr << "Invalid data size" << std::endl;
        return false;
    }

    Elapse el("Filter FFT", 16);
    Transpose1DC<<<DIV(num_of_filters, GRP), GRP>>>(*filter, *filterFFT, num_of_channels, filter_size, samples_per_channel_padd, 1);
    el.Stamp("Transpose Filter");

    FFT::Dispatch(true, filterFFT, num_of_channels);
    el.Stamp("Filter FFT");
    FillDFTWeights<<<DIV(num_of_channels * num_of_channels, GRP), GRP>>>(*weightsDFT, num_of_channels);
    el.Stamp("Weights DFT");

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool SP::SupermanProcess()
{
    int errors = 0, k = 0, L = 30;
    Elapse el("SupermanProcess", 16);

    for (k = 0; k < 1000; ++k)
    {
        el.Loop("anis", true, k < L);
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
        //        currT.swap(out);
        //        MultiDFT<<<DIV(num_of_samples, GRP), GRP>>>(*out, *currT, *weightsDFT, samples_per_channel, num_of_channels);
        FFT::Dispatch(true, out, samples_per_channel);
        el.Stamp("Multi FFT", k < L);
        AbsMag<<<DIV(num_of_samples, GRP), GRP>>>(*out, *currAbs, num_of_samples);
        el.Stamp("AbsMag", k < L);
        convolve1DFirIL<<<DIV(num_of_samples, GRP), GRP>>>(*abs_prev, *currAbs, *currFir, samples_per_channel, num_of_channels);
        el.Stamp("FirIL", k < L);
#endif
        el.Loop("anis", false, k < L);
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
bool SP::SpidermanInit()
{
    for (int j = 0; j < 6; ++j)
    {
        m_windowSizes[j] = 1 << m_SF[j];
        m_hopSizes[j] = m_windowSizes[j] - int(m_windowSizes[j] * OVERLAP);
    }
    for (int i = 0; i < 4; ++i)
    {
        const int signalSize = DIV(CHUNK_SIZE, m_downSamples[i]);
        m_signalSizes[i] = signalSize;

        decimate4[i].reallocate(signalSize);
        dechirp4[i].reallocate(signalSize * 6 * 2);

        for (int j = 0; j < 6; ++j)
        {
            const int windowSize = m_windowSizes[j];
            const int numOfWnds = DIV(signalSize, windowSize) * windowSize / m_hopSizes[j];
            m_numOfWindows[j * 4 + i] = numOfWnds;
        }
    }
    for (int j = 0; j < 6; ++j)
    {
        for (int i = 0; i < 4; ++i)
        {
            m_windowsOffset[j * 4 + i] = m_totNumOfWindows[j];
            m_totNumOfWindows[j] += m_numOfWindows[j * 4 + i];
        }
    }
    for (int j = 0; j < 6; ++j)
    {
        overlapSignal6[j].reallocate(m_windowSizes[j] * m_totNumOfWindows[j]);
        stats6[j].reallocate(m_totNumOfWindows[j]);
    }

    channChunk.reallocate(CHUNK_SIZE * 2);

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool SP::SpidermanLoad(int channel)
{
    //    raw.ReadFromFile(workDir + "t/raw.32fc");
    //    Utils::FromFile(workDir + "t/ddc.32fc", ddc1);
    //    /**
    //     * if (fd > fs /2)
    //     *      fd = - (fd % (fs / 2))
    //     */
    //    double fs = 8.5e6;
    //    double fd = 6.5e6;
    //    double invFsd = (1.0 / fs) * fd;
    //
    ////    0.0117747 - 0.0179254i
    ////    0.0119519 - 0.0179124i
    //    float err;
    //    int errors= 0;
    //    for(int idx = 0; idx < 10; ++idx){
    //        double ts = double(idx) * (fd / fs);
    //        float2 s = raw[idx], b = ddc1[idx];
    //        double p = -2.0 * M_PI * ts;
    //        double2 e = {cos(p), sin(p)};
    ////        float2 e = CW(ts);
    //
    //        double2 ad = cuCmul(e, {s.x, s.y});
    //        float2 a = {ad.x, ad.y};
    //        if(false == CMP(a,b,&err))
    //            ++errors;
    //    }
    //    freqShift<<<DIV(raw.size(), GRP), GRP>>>(*raw, raw.size(), invFsd);
    //    sync();

    //    auto err = Compare(raw, 0, ddc1);
    //    std::cout << "ERRRR = " << err << std::endl;

    fir2.ReadFromFile(spdDir + "fir2.32f");
    fir4.ReadFromFile(spdDir + "fir4.32f");
    fir8.ReadFromFile(spdDir + "fir8.32f");
    if (fir2.size() <= 0 || fir4.size() <= 0 || fir8.size() <= 0)
        return false;

    std::string chanDir = spdDir + std::to_string(channel) + "/";
    chann1.ReadFromFile(chanDir + "input.32fc");
    if (chann1.size() <= 0)
        return false;

    for (int i = 0; i < 24; ++i)
    {
        const int configId = i + 1;
        std::string config = std::to_string(configId);

        bool rv = true;
        rv = rv && Utils::FromFile(chanDir + "decimate" + config + ".32fc", decimate24[i]);
        rv = rv && Utils::FromFile(chanDir + "dechirp" + config + ".32fc", dechirp24[i]);
        rv = rv && Utils::FromFile(chanDir + "stft" + config + ".32fc", stft24[i]);

        if (false == rv)
            return false;

        dechirp24res[i].reallocate(dechirp24[i].size());
        stft24res[i].reallocate(stft24[i].size());
    }
    for (int i = 0; i < 4; ++i)
        decimate4res[i].reallocate(decimate24[i].size());

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
gbuffer<float> *SP::GetFirFilter(const int downSample)
{
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
    return pFir;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool SP::SpidermanSingleProcess(int configId)
{
    auto config = std::to_string(configId);
    const int configIdx = configId - 1;
    const int SF = (configIdx / 4) + 7;
    const int windowSize = 1 << SF;
    const int overlapSize = (int)(windowSize * OVERLAP);
    const int hopSize = windowSize - overlapSize;
    double MFS = 1.0 / (1 << SF);
    const int downSample = 1 << (3 - configIdx % 4);

    const int signalSize = DIV(chann1.size(), downSample);
    const int m_numOfWindows = DIV(signalSize, windowSize) * windowSize / hopSize;

    int k = 0, L = 30, LOOPS = 1;
    Elapse el("Single ZYX config", 16);

    signalDown.resize(signalSize);
    dechirp.resize(signalSize);
    overlapSignal.resize(m_numOfWindows * windowSize);
    //    stats.m_host = true;
    stats.resize(m_numOfWindows);
    int errors = 0;

    for (k = 0; k < LOOPS; ++k)
    {
        el.Loop("Single Spiderman", true, k < L);
        gbuffer<float> *pFir = GetFirFilter(downSample);
        if (pFir)
        {
            auto &fir = *pFir;
            const int filterSize = (int)fir.size();

            convolve1DDown<<<DIV(signalSize, GRP), GRP>>>(
                *chann1,
                *fir,
                *signalDown,
                (int)chann1.size() - filterSize,
                signalSize,
                filterSize,
                downSample,
                0);
        }
        else
        {
            Copy(*signalDown, *chann1, signalSize);
        }
        el.Stamp("Decimate", k < L);

        const int dsize = (int)dechirp.size();
        DeChirp<<<DIV(dsize, GRP), GRP>>>(*signalDown, *dechirp, dsize, MFS, 0, windowSize << 1);
        el.Stamp("DeChirp", k < L);

        dim3 grid(DIV(windowSize, GRP), m_numOfWindows);
        applyOverlapSignal<<<grid, GRP>>>(*dechirp, *overlapSignal, (int)dechirp.size(), windowSize, hopSize, m_numOfWindows);
        el.Stamp("Apply Window");

        FFT::Dispatch(true, overlapSignal, m_numOfWindows);
        fftShift<<<grid, GRP>>>(*overlapSignal, windowSize, m_numOfWindows);
        el.Stamp("STFT");

        calcStatsPerWindow<<<DIV(m_numOfWindows, GRP), GRP>>>(*stats, *overlapSignal, windowSize, m_numOfWindows);
        el.Stamp("Stats");

        el.Loop("Single Spiderman", false, k < L);
    }
    sync();
    errors = Compare(signalDown, 0, decimate24[configIdx]);
    std::cout << "Decimate Chann [" + config + "] Errors: " << errors << std::endl;

    errors = Compare(dechirp, 0, dechirp24[configIdx]);
    std::cout << "DeChirp [" + config + "] Errors: " << errors << std::endl;

    errors = Compare(overlapSignal, 0, stft24[configIdx]);
    std::cout << "STFT Chann [" + config + "] Errors: " << errors << std::endl;

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool SP::SpidermanSingleSplitProcess(int configId)
{
    auto config = std::to_string(configId);
    const int configIdx = configId - 1;
    const int SF = (configIdx / 4) + 7;
    const int windowSize = 1 << SF;
    const int overlapSize = (int)(windowSize * OVERLAP);
    const int hopSize = windowSize - overlapSize;
    double MFS = 1.0 / (1 << SF);
    const int downSample = 1 << (3 - configIdx % 4);
    const int chunkCount = (int)chann1.size() / CHUNK_SIZE;
    std::cout << "--------------------------------------------------------\n";
    int k = 0, L = 30, LOOPS = 1;
    Elapse el("Spiderman config chunk", 16);

    const int signalSize = DIV(CHUNK_SIZE, downSample);
    const int m_numOfWindows = DIV(signalSize, windowSize) * windowSize / hopSize;
    const int stftSize = m_numOfWindows * windowSize;

    channChunk.reallocate(CHUNK_SIZE * 2);
    signalDown.reallocate(signalSize);
    dechirp.reallocate(signalSize * 2);
    overlapSignal.reallocate(stftSize);
    stats.reallocate(m_numOfWindows);

    Zero(channChunk.p(0), CHUNK_SIZE);
    Zero(dechirp.p(0), signalSize);

    int curr = 0, prev = 0;
    for (int chunkIdx = 0; chunkIdx < chunkCount; ++chunkIdx)
    {
        const int chunkOff = chunkIdx * CHUNK_SIZE;
        prev = curr;
        curr = (curr + 1) % 2;

        Copy(channChunk.p(curr * CHUNK_SIZE), chann1.p(chunkOff), CHUNK_SIZE);

        int errors = 0;
        for (k = 0; k < LOOPS; ++k)
        {
            el.Loop("Single Spiderman", true, k < L);
            gbuffer<float> *pFir = GetFirFilter(downSample);
            if (pFir)
            {
                auto &fir = *pFir;
                const int filterSize = (int)fir.size();

                convolve1DDownPrev<<<DIV(signalSize, GRP), GRP>>>(
                    channChunk.p(prev * CHUNK_SIZE),
                    channChunk.p(curr * CHUNK_SIZE),
                    *fir, *signalDown,
                    CHUNK_SIZE,
                    signalSize,
                    filterSize,
                    downSample,
                    0);
            }
            else
            {
                Copy(*signalDown, channChunk.p(curr * CHUNK_SIZE), signalSize);
            }
            el.Stamp("Decimate", k < L);

            DeChirp<<<DIV(signalSize, GRP), GRP>>>(*signalDown, dechirp.p(curr * signalSize), signalSize, MFS, chunkIdx * signalSize, windowSize << 1);
            el.Stamp("DeChirp", k < L);

            dim3 grid(DIV(windowSize, GRP), m_numOfWindows);
            applyOverlapSignalPrev<<<grid, GRP>>>(dechirp.p(prev * signalSize), dechirp.p(curr * signalSize), *overlapSignal, signalSize, windowSize, hopSize, m_numOfWindows);
            el.Stamp("Apply Window");

            FFT::Dispatch(true, overlapSignal, m_numOfWindows);
            fftShift<<<grid, GRP>>>(*overlapSignal, windowSize, m_numOfWindows);
            el.Stamp("STFT");

            calcStatsPerWindow<<<DIV(m_numOfWindows, GRP), GRP>>>(*stats, *overlapSignal, windowSize, m_numOfWindows);
            el.Stamp("Stats");

            el.Loop("Single Spiderman", false, k < L);
        }
        sync();
        errors = Compare(signalDown, 0, decimate24[configIdx], chunkIdx * signalSize, signalSize);
        if (errors > 0)
            std::cout << "Decimate Chann [" + config + "] Chunk=" << chunkIdx << " Errors: " << errors << std::endl;

        dechirp.RefreshDown();
        errors = Compare(dechirp.h(curr * signalSize), 0, dechirp24[configIdx].data(), chunkIdx * signalSize, signalSize);
        if (errors > 0)
            std::cout << "DeChirp [" + config + "] Chunk=" << chunkIdx << " Errors: " << errors << std::endl;

        errors = Compare(overlapSignal, 0, stft24[configIdx], chunkIdx * stftSize, stftSize);
        if (errors > 0)
            std::cout << "STFT Chann [" + config + "] Chunk=" << chunkIdx << " Errors: " << errors << std::endl;
        //        break;
    }
    std::cout << "----------\t" << config << "\t----------------------------------------\n";

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool SP::SpidermanBatchSplitProcess()
{
    int k = 0, L = 30, LOOPS = 1;
    Elapse el("SpidermanBatchSplitProcess", 16);

    const int chunkCount = (int)chann1.size() / CHUNK_SIZE;

    for (k = 0; k < LOOPS; ++k)
    {
        Zero(channChunk.p(0), CHUNK_SIZE);
        for (int i = 0; i < 4; ++i)
            Zero(dechirp4[i].p(0), m_signalSizes[i] * 6);

        el.Loop("Batch Spiderman", true, k < L);
        int curr = 0, prev = 0;
        for (int chunkIdx = 0; chunkIdx < chunkCount; ++chunkIdx)
        {
            const int chunkOff = chunkIdx * CHUNK_SIZE;
            prev = curr;
            curr = (curr + 1) % 2;
            Copy(channChunk.p(curr * CHUNK_SIZE), chann1.p(chunkOff), CHUNK_SIZE);

            el.Loop("Batch Spiderman_chunk", true, k < L);

            for (int i = 0; i < 4; ++i)
            {
                const int downSample = m_downSamples[i];

                gbuffer<float> *pFir = GetFirFilter(downSample);
                if (pFir)
                {
                    auto &fir = *pFir;
                    const int filterSize = (int)fir.size();
                    const int sharedMemSize = DIV(filterSize, GRP) * GRP * sizeof(float);

                    convolve1DDownPrev<<<DIV(m_signalSizes[i], GRP), GRP, sharedMemSize>>>(
                        channChunk.p(prev * CHUNK_SIZE),
                        channChunk.p(curr * CHUNK_SIZE),
                        *fir, *decimate4[i],
                        CHUNK_SIZE,
                        m_signalSizes[i],
                        filterSize,
                        downSample,
                        0);
                }
                else
                {
                    Copy(*decimate4[i], channChunk.p(curr * CHUNK_SIZE), m_signalSizes[i]);
                }
            }
            el.Stamp("DecimateAll", k < L);

            for (int i = 0; i < 4; ++i)
            {
                const int signalSize = m_signalSizes[i];
                for (int j = 0; j < 6; ++j)
                {
                    double MFS = 1.0 / (1 << m_SF[j]);
                    const int offset = curr * signalSize * 6 + j * signalSize;
                    DeChirp<<<DIV(signalSize, GRP), GRP>>>(
                        *decimate4[i],
                        dechirp4[i].p(offset),
                        signalSize,
                        MFS,
                        chunkIdx * signalSize,
                        m_windowSizes[j] << 1);
                }
            }
            el.Stamp("DeChirpAll", k < L);

            for (int j = 0; j < 6; ++j)
            {
                const int hopSize = m_hopSizes[j];
                const int windowSize = m_windowSizes[j];
                for (int i = 0; i < 4; ++i)
                {
                    const int signalSize = m_signalSizes[i];
                    const int prevOffset = prev * signalSize * 6 + j * signalSize;
                    const int currOffset = curr * signalSize * 6 + j * signalSize;
                    const int numOfWnds = m_numOfWindows[j * 4 + i];
                    const int windowOffset = m_windowsOffset[j * 4 + i] * windowSize;
                    dim3 grid(DIV(windowSize, GRP), numOfWnds);
                    applyOverlapSignalPrev<<<grid, GRP>>>(
                        dechirp4[i].p(prevOffset),
                        dechirp4[i].p(currOffset),
                        overlapSignal6[j].p(windowOffset),
                        signalSize,
                        windowSize,
                        hopSize,
                        numOfWnds);
                }
            }
            el.Stamp("Apply Window All", k < L);

            for (int j = 0; j < 6; ++j)
            {
                const int windowSize = m_windowSizes[j];
                const int numOfWnds = m_totNumOfWindows[j];
                dim3 grid(DIV(windowSize, GRP), numOfWnds);
                FFT::Dispatch(true, overlapSignal6[j], numOfWnds);
                fftShift<<<grid, GRP>>>(*overlapSignal6[j], windowSize, numOfWnds);
            }
            el.Stamp("STFT All", k < L);

            for (int j = 0; j < 6; ++j)
            {
                calcStatsPerWindow<<<DIV(m_totNumOfWindows[j], GRP), GRP>>>(
                    *stats6[j],
                    *overlapSignal6[j],
                    m_windowSizes[j],
                    m_totNumOfWindows[j]);
            }
            el.Stamp("Stats All", k < L);

            el.Loop("Batch Spiderman_chunk", false, k < L);

            sync();
            // save chunks one by one to compare later with results
            for (int i = 0; i < 4; ++i)
            {
                Copy(decimate4res[i].p(chunkIdx * m_signalSizes[i]), decimate4[i].p(), m_signalSizes[i]);
                for (int j = 0; j < 6; ++j)
                {
                    const int configIdx = j * 4 + i;
                    const int signalSize = m_signalSizes[i];
                    const int offset = curr * signalSize * 6 + j * signalSize;
                    Copy(dechirp24res[configIdx].p(chunkIdx * signalSize), dechirp4[i].p(offset), signalSize);

                    const int windowSize = m_windowSizes[j];
                    const int windowOffset = m_windowsOffset[configIdx] * windowSize;
                    const int stftSize = windowSize * m_numOfWindows[configIdx];
                    Copy(stft24res[configIdx].p(chunkIdx * stftSize), overlapSignal6[j].p(windowOffset), stftSize);
                }
            }
// #define FIND_ERROR_PER_CHUNK
#ifdef FIND_ERROR_PER_CHUNK
            for (int i = 0; i < 4; ++i)
            {
                decimate4[i].RefreshDown();
                dechirp4[i].RefreshDown();
            }
            for (int j = 0; j < 6; ++j)
                overlapSignal6[j].RefreshDown();

            int errors = 0;
            for (int configIdx = 0; configIdx < 24; ++configIdx)
            {
                const int i = configIdx % 4;
                errors = Compare(decimate4[i], 0, decimate24[configIdx], chunkIdx * m_signalSizes[i], m_signalSizes[i]);
            }
            if (errors > 0)
                std::cout << "Decimate All Errors: " << errors << std::endl;

            errors = 0;
            for (int configIdx = 0; configIdx < 24; ++configIdx)
            {
                const int i = configIdx % 4;
                const int j = (configIdx / 4);
                const int signalSize = m_signalSizes[i];
                const int offset = curr * signalSize * 6 + j * signalSize;
                errors += Compare(dechirp4[i].h(offset), 0, dechirp24[configIdx].data(), chunkIdx * signalSize, signalSize);
            }
            if (errors > 0)
                std::cout << "DeChirp All Errors: " << errors << std::endl;

            errors = 0;
            for (int configIdx = 0; configIdx < 24; ++configIdx)
            {
                const int j = (configIdx / 4);
                const int windowSize = m_windowSizes[j];
                const int windowOffset = m_windowsOffset[configIdx] * windowSize;
                const int stftSize = windowSize * m_numOfWindows[configIdx];
                errors += Compare(overlapSignal6[j], windowOffset, stft24[configIdx], chunkIdx * stftSize, stftSize);
            }
            if (errors > 0)
                std::cout << "STFT All Errors: " << errors << std::endl;
#endif
        }
        el.Loop("Batch Spiderman", false, k < L);
    }
    int errors = 0;
    for (int i = 0; i < 24; ++i)
        errors += Compare(decimate4res[i % 4], 0, decimate24[i], 0, m_signalSizes[i % 4] * chunkCount);
    std::cout << "Decimate All Errors: " << errors << std::endl;

    errors = 0;
    for (int i = 0; i < 24; ++i)
        errors += Compare(dechirp24res[i], 0, dechirp24[i], 0, m_signalSizes[i % 4] * chunkCount);
    std::cout << "Dechirp All Errors: " << errors << std::endl;

    errors = 0;
    for (int i = 0; i < 24; ++i)
        errors += Compare(stft24res[i], 0, stft24[i], 0, m_windowSizes[i / 4] * m_numOfWindows[i] * chunkCount);
    std::cout << "STFT All Errors: " << errors << std::endl;

    return true;
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
__device__ float2 DFTW(int k, int n)
{
    float a = -2.0f * M_PI * k / n;
    return {cosf(a), sinf(a)};
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__host__ __device__ float2 CW(double t)
{
    double p = -2.0 * M_PI * t;
    return {(float)cos(p), (float)sin(p)};
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
__global__ void DeChirp(const float2 *input, float2 *output, size_t size, double MFS, int chirpOffset, int chirpSize)
{
    int idx = getI();
    if (idx >= size)
        return;

    //    int cidx = (idx + chirpOffset)%chirpSize;
    int cidx = (idx + chirpOffset); // no need to modulo by chirp size, will auto create because of sin/cos

    float2 chirp_sample = ChirpK(cidx, MFS);
    float2 sample = input[idx];

    output[idx] = cuCmulf(chirp_sample, sample);
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
__global__ void convolve1DDownPrev(const float2 *prev, const float2 *curr, const float *filter, float2 *output, const int size, const int downSize, const int filterSize, const int downSample, const int offset)
{
    extern __shared__ float shared_filter[];
    int val_per_thread = DIV(filterSize, blockDim.x);
    for (int i = 0; i < val_per_thread; ++i)
    {
        int tid = threadIdx.x + i * blockDim.x;
        if (tid < filterSize)
            shared_filter[tid] = filter[tid];
    }
    __syncthreads();

    int idx = getI();
    if (idx >= downSize)
        return;

    const int I = (idx + offset) * downSample;

    float2 res = {0.f, 0.f};
    for (int f = 0, s = I + 1 - filterSize; f < filterSize; ++f, ++s)
    {
        const float filter_val = shared_filter[f];
        if (s < size)
        {
            if (s >= 0)
                res += curr[s] * filter_val;
            else
                res += prev[s + size] * filter_val;
        }
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
    float2 *outputSignal,      // output size = windowSize * m_numOfWindows
    int signalSize,            // unpadded signal size
    int windowSize,
    int hopSize,
    int m_numOfWindows)
{
    const int idx = getI(); // [0, windowSize)
    const int jdx = getJ(); // [0, m_numOfWindows)
    if (idx >= windowSize || jdx >= m_numOfWindows)
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
__global__ void applyOverlapSignalPrev(
    const float2 *prev,
    const float2 *curr,
    float2 *outputSignal,
    int signalSize,
    int windowSize,
    int hopSize,
    int m_numOfWindows)
{
    const int idx = getI(); // [0, windowSize)
    const int jdx = getJ(); // [0, m_numOfWindows)
    if (idx >= windowSize || jdx >= m_numOfWindows)
        return;

    const int overlapSize = windowSize - hopSize;
    const int ibase = jdx * hopSize + idx - overlapSize;
    const int obase = jdx * windowSize + idx;

    if (0 <= ibase && ibase < signalSize)
        outputSignal[obase] = curr[ibase];
    else if (ibase < signalSize)
        outputSignal[obase] = prev[ibase + signalSize];
    else
        outputSignal[obase] = {0.f, 0.f};
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
__global__ void freqShift(float2 *buffer, int size, double invFS) // InvFS = (1 / FS) * df
{
    int idx = getI();
    if (idx >= size)
        return;

    //    matlab
    //    ts = 1/fs;
    //    N = length(x);
    //    tvec = (0:ts:(N-1)*ts).';
    //
    //    y = x.*exp(2*pi*1j*df.*tvec);
    double fs = 8.5e6;
    double fd = 6.5e6;

    double ts = double(idx) * (fd / fs);
    //    if (idx <= 10)
    //        printf("Idx: %d, ts: %.9f\n", idx, ts);

    // double tdf = idx * invFS; // tvec(i) * df
    float2 e = CW(ts), s = buffer[idx];

    float2 d = cuCmulf(e, s);
    //    float2 d = e * s;
    if (idx <= 10)
        printf("Idx: %d, ts: %.9f, sx: %.9f, sy: %.9f, dx: %.9f, dy: %.9f\n", idx, ts, s.x, s.y, d.x, d.y);

    buffer[idx] = d;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void fftShift(float2 *buffer, int windowSize, int m_numOfWindows)
{
    int idx = getI(); // [0, windowSize/2)
    int jdx = getJ(); // [0, m_numOfWindows)
    if (idx >= windowSize / 2 || jdx >= m_numOfWindows)
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
__global__ void calcStatsPerWindow(float4 *output, const float2 *input, int windowSize, int m_numOfWindows)
{
    int idx = getI(); // [0, m_numOfWindows)
    if (idx >= m_numOfWindows)
        return;

    int maxIndex = -1;
    float2 smp;
    float tot = 0.f, maxVal = 0.f, val;
    for (int i = 0; i < windowSize; ++i)
    {
        smp = input[idx * windowSize + i];
        val = cuCabsf(smp);
        tot += val;
        if (maxVal < val)
        {
            maxVal = val;
            maxIndex = i;
        }
    }

    float avg = tot / windowSize;
    float mavg = avg / maxVal;

    output[idx] = {maxVal, avg, mavg, float(maxIndex)};
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(cu);
