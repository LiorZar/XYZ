#pragma once
#include "cu/GPU.h"
#include "cu/FFT.h"

NAMESPACE_BEGIN(cu);

const int GRP = 256;
const int num_of_channels = 20;
const int samples_per_channel = 20000;
const int filter_size = 500;
const int window_size = 1024;
const int overlap_size = 512;
const int hop_size = window_size - overlap_size;
// const int samples_per_channel_padd = samples_per_channel + filter_size - 1;
const int samples_per_channel_padd = (int)FFT::NextPow2(samples_per_channel + filter_size * 2 - 1);
const int num_of_samples = num_of_channels * samples_per_channel;
const int num_of_filters = num_of_channels * filter_size;
const int num_of_samples_padd = num_of_channels * samples_per_channel_padd;
const int num_of_windows = (samples_per_channel - window_size) / hop_size + 1;

class SP
{
public:
    SP();
    ~SP() = default;

public:
    bool ToCSV();
    bool Init();
    bool Process();
    bool STFT();
    bool MinMax();

private:
    // for gpu
    gbuffer<int> globals;
    gbuffer<float> firFilter, currAbs, currFir;
    gbuffer<float2> filterFFT;
    gbuffer<float2> prevT, currT, signal1, signal1out;
    gbuffer<float> hanningWindow, hammingWindow;

    // for testing
    gbuffer<float> filter, abs_out, abs_prev, fir_out;
    gbuffer<float2> fft_out;
    gbuffer<float2> prev, curr, out, result;
};

NAMESPACE_END(cu);