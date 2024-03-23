#pragma once
#include "cu/GPU.h"
#include "cu/FFT.h"

NAMESPACE_BEGIN(cu);

const int GRP = 128;
const int num_of_channels = 20;
const int samples_per_channel = 20000;
const int filter_size = 500;
const int samples_per_channel_padd = (int)FFT::NextPow2(samples_per_channel + filter_size * 2 - 1);
const int num_of_samples = num_of_channels * samples_per_channel;
const int num_of_filters = num_of_channels * filter_size;
const int num_of_samples_padd = num_of_channels * samples_per_channel_padd;

class SP
{
public:
    SP();
    ~SP() = default;

public:
    bool SupermanInit();
    bool SupermanProcess();

public:
    bool SpidermanInit();
    bool SpidermanLoad(int channel);
    bool SpidermanSingleProcess(int configId);
    bool SpidermanSingleSplitProcess(int configId);
    bool SpidermanBatchSplitProcess();

private:
    gbuffer<float> *GetFirFilter(const int downSample);

private:
    // for gpu
    gbuffer<int> globals;
    gbuffer<float> firFilter, currAbs, currFir;
    gbuffer<float2> filterFFT, weightsDFT;
    gbuffer<float2> prevT, currT;

    gbuffer<float> fir2, fir4, fir8;
    gbuffer<float2> raw, ddc, chann1;
    gbuffer<float2> channChunk, dechirp, signalDown, overlapSignal;
    gbuffer<float4> stats;

    gbuffer<float2> decimate4[4], dechirp4[4], overlapSignal6[6];
    gbuffer<float4> stats6[6];

    int m_signalSizes[4];
    int m_downSamples[4] = {8, 4, 2, 1};
    int m_windowSizes[6], m_hopSizes[6], m_numOfWindows[24], m_windowsOffset[24], m_totNumOfWindows[6] = {0, 0, 0, 0, 0, 0};
    int m_SF[6] = {7, 8, 9, 10, 11, 12};

    // for testing
    gbuffer<float> filter, abs_out, abs_prev, fir_out;
    gbuffer<float2> fft_out;
    gbuffer<float2> prev, curr, out, tout, result;

    // for check validity
    std::vector<float2> ddc1;
    std::vector<float2> decimate24[24], dechirp24[24], stft24[24];
    gbuffer<float2> decimate4res[4], dechirp24res[24], stft24res[24];

    std::string workDir, supDir, spdDir;
};

NAMESPACE_END(cu);