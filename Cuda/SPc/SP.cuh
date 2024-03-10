#pragma once
#include "cu/GPU.h"
#include "cu/FFT.h"

NAMESPACE_BEGIN(cu);

const int GRP = 128;
const int num_of_channels = 20;
const int samples_per_channel = 20000;
const int filter_size = 500;
// const int samples_per_channel_padd = samples_per_channel + filter_size - 1;
const int samples_per_channel_padd = (int)FFT::NextPow2(samples_per_channel + filter_size * 2 - 1);
const int num_of_samples = num_of_channels * samples_per_channel;
const int num_of_filters = num_of_channels * filter_size;
const int num_of_samples_padd = num_of_channels * samples_per_channel_padd;

const int test_size = 4096;
const int window_size = 256;
const int overlap_size = window_size / 2;
const int hop_size = window_size - overlap_size;
const int num_of_windows = (test_size - window_size) / hop_size + 1;

class SP
{
public:
    SP();
    ~SP() = default;

public:
    bool Init();
    bool XYZProcess();
    bool SingleZYXProcess(int channel_id);
    bool MinMax();

private:
    bool Decimate(int channel);
    bool Chrip(int channel);
    bool STFT(int channel);

    bool ZYXLoadFile(int channel, int &downSample, int &signalSize, int &windowSize, int &hopSize, int &numOfWindows, int &SF, bool saveImage = false);

private:
    // for gpu
    gbuffer<int> globals;
    gbuffer<float> firFilter, currAbs, currFir;
    gbuffer<float2> filterFFT, weightsDFT;
    gbuffer<float2> prevT, currT, signal1, signal1out, signal1outT;

    gbuffer<float> hanningWindow, hammingWindow, onesWindow;
    gbuffer<float> fir2, fir4, fir8;
    gbuffer<float2> chann1, chann1Padd, chann1FFT, decimate1, chirp1, dechirp1, stft1;
    gbuffer<float2> dechirp, signalDown, overlapSignal;

    // for testing
    gbuffer<float> filter, abs_out, abs_prev, fir_out;
    gbuffer<float2> fft_out;
    gbuffer<float2> prev, curr, out, tout, result;

    std::string workDir;
};

NAMESPACE_END(cu);