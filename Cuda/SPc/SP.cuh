#pragma once
#include "cu/GPU.h"
#include "cu/FFT.h"

NAMESPACE_BEGIN(cu);

const int GRP = 256;
const int num_of_channels = 20;
const int samples_per_channel = 20000;
const int filter_size = 500;
const int window_size = 1024;
const int hop_size = 512;
// const int samples_per_channel_padd = samples_per_channel + filter_size - 1;
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
    bool Init();
    bool Process();

private:
    // for gpu
    gbuffer<float> firFilter, currAbs, currFir;
    gbuffer<float2> filterFFT;
    gbuffer<float2> prevT, currT;
    gbuffer<float> hanningWindow, hammingWindow;

    // for testing
    gbuffer<float> filter, abs_out, abs_prev, fir_out;
    gbuffer<float2> fft_out;
    gbuffer<float2> prev, curr, out, result;
};

NAMESPACE_END(cu);