#include "cl/defines.h"
#include "cl/Program.h"
#include "cl/Buffer.hpp"
#include "cl/FFT.h"
#include "cl/Elapse.hpp"

const int GRP = 256;
const int num_of_channels = 20;
const int samples_per_channel = 20000;
const int filter_size = 500;
// const int samples_per_channel_padd = samples_per_channel + filter_size - 1;
const int samples_per_channel_padd = FFT::NextPow2(samples_per_channel + filter_size - 1);
const int num_of_samples = num_of_channels * samples_per_channel;
const int num_of_filters = num_of_channels * filter_size;
const int num_of_samples_padd = num_of_channels * samples_per_channel_padd;

int Compare(const std::vector<std::complex<float>> &a, const std::vector<std::complex<float>> &b, size_t size)
{
    int errors = 0;
    for (auto i = 0U; i < size; ++i)
    {
        auto diff = a[i] - b[i];
        if (std::abs(diff) > 1e-6)
        {
            // std::cerr << "Error at " << i << ": " << a[i] << " != " << b[i] << std::endl;
            ++errors;
        }
    }
    return errors;
}

std::vector<std::complex<float>> Tone(float frequency, int sampleRate, float duration, size_t size)
{
    int numSamples = static_cast<int>(duration * sampleRate); // Total number of samples
    std::vector<std::complex<float>> signal(std::max<size_t>(size, numSamples));

    // Generate each sample of the sine wave
    for (int i = 0; i < numSamples; ++i)
    {
        float t = static_cast<float>(i) / sampleRate;                       // Current time in seconds for this sample
        signal[i].real(sin(2.0f * 3.14159265358979323846 * frequency * t)); // Calculate the sine value
    }

    return signal;
}
int main()
{

    Elapse el("Main");
    std::cout << "----------------------------------------------------------\n";
    Context::getInstance();
    FFT::getInstance();
    Program program("../../src/kernels/main.cl");

    // Buffer<float> filter, filter1(num_of_filters);
    Buffer<float> filter, filter1(num_of_samples_padd);
    Buffer<std::complex<float>> curr1[2], prev1[2], out1[2], currFFT[2], filter2(num_of_samples_padd), filterFFT(num_of_samples_padd);
    Buffer<std::complex<float>> curr, prev, out(num_of_samples), result;

    curr.ReadFromFile("../../../data/0/curr.bin");
    prev.ReadFromFile("../../../data/0/prev.bin");
    filter.ReadFromFile("../../../data/0/filter.bin");
    result.ReadFromFile("../../../data/0/out.bin", false);
    if (filter.size() != num_of_filters || curr.size() != num_of_samples || prev.size() != num_of_samples || result.size() != num_of_samples)
    {
        std::cerr << "Invalid data size" << std::endl;
        return 1;
    }
    el.Stamp("Data loaded");

    for (int i = 0; i < 2; ++i)
    {
        curr1[i].Resize(num_of_samples_padd);
        prev1[i].Resize(num_of_samples_padd);
        out1[i].Resize(num_of_samples_padd);
    }
    el.Stamp("Data allocated");
    program.Dispatch1D("Transpose1DC", num_of_filters, GRP, *filter, *filter2, num_of_channels, filter_size, samples_per_channel_padd);
    for (int i = 0; i < num_of_channels; ++i)
        FFT::Dispatch(true, filter2, filterFFT, i * samples_per_channel_padd, samples_per_channel_padd);
    // // FFT::Dispatch(true, *filter2, *filterFFT, num_of_filters);
    // // FFT::Dispatch(true, filter2, filterFFT, samples_per_channel_padd, samples_per_channel_padd);
    el.Stamp("FFT filter");
    filterFFT.Download();

    {
        Buffer<std::complex<float>> filter(samples_per_channel_padd), buffer, buffer1(samples_per_channel_padd), buffer2(samples_per_channel_padd), res1(samples_per_channel_padd), res2(samples_per_channel_padd);
        filter2.Download();
        for (int i = 0; i < filter_size; ++i)
            filter[i] = filter2[i];
        filter.Upload();
        buffer.host() = Tone(440, 20000, 1, samples_per_channel_padd);
        buffer.Upload();

        for (int i = 0; i < 20000; ++i)
        {
            std::complex<float> sum = 0;
            for (int j = 0, s = i + 1 - 500; j < 500; ++j, ++s)
            {
                if (s < 0)
                    continue;
                if (s >= 20000)
                    continue;
                sum += buffer[s] * filter[j];
            }
            res1[i] = sum;
        }
        res1.Upload();
        FFT::Dispatch(true, res1, res2);
        res2.Download();
        FFT::Dispatch(true, filter);
        FFT::Dispatch(true, buffer, buffer1);
        filter.Download();
        buffer1.Download();
        for (int i = 0; i < samples_per_channel_padd; ++i)
        {
            auto n = filter[i] * buffer1[i];
            ;
            auto x = res2[i];
            auto diff = n - x;
            res2[i] = n;
        }
        res2.Upload();
        FFT::Dispatch(false, res2);
        res2.Download();
        for (int i = 0; i < 1000; ++i)
        {
            auto diff = res1[i] - res2[i];
            auto d = diff.real() * diff.real() + diff.imag() * diff.imag();
            if (d > 1e-6)
                std::cerr << "Error at " << i << ": " << res1[i] << " != " << res2[i] << std::endl;
        }
        FFT::Dispatch(false, buffer1, buffer2);
        buffer2.Download();
        for (int i = 0; i < buffer.size(); ++i)
        {
            auto diff = buffer[i] - buffer2[i];
            if (std::abs(diff) > 1e-6)
                std::cerr << "Error at " << i << ": " << buffer[i] << " != " << buffer2[i] << std::endl;
        }
    }

    // int tt = program.Dispatch1D("convolve2D", num_of_samples, GRP, *prev, *curr, *filter, *out, num_of_samples, filter_size, num_of_channels);
    // int t0 = program.Dispatch1D("Transpose2DPalanar", num_of_samples, GRP, *curr, *currx, *curry, num_of_channels, samples_per_channel);
    // int tt = program.Dispatch1D("Transpose1D", num_of_filters, GRP, *filter, *filter1, num_of_channels, filter_size, filter_size);
    int tt = program.Dispatch1D("Transpose1D", num_of_filters, GRP, *filter, *filter1, num_of_channels, filter_size, samples_per_channel_padd);
    // int t0 = program.Dispatch1D("Transpose2DPalanarC", num_of_samples, GRP, *curr, *curr1[0], *curr1[1], num_of_channels, samples_per_channel, samples_per_channel);
    // int t1 = program.Dispatch1D("Transpose2DPalanarC", num_of_samples, GRP, *prev, *prev1[0], *prev1[1], num_of_channels, samples_per_channel, samples_per_channel);
    int t0 = program.Dispatch1D("Transpose2DPalanarC", num_of_samples, GRP, *curr, *curr1[0], *curr1[1], num_of_channels, samples_per_channel, samples_per_channel_padd);
    int t1 = program.Dispatch1D("Transpose2DPalanarC", num_of_samples, GRP, *prev, *prev1[0], *prev1[1], num_of_channels, samples_per_channel, samples_per_channel_padd);
    el.Stamp("Transpose2DPalanar");
    // filter1.Download();
    // curr1[0].Download();
    // curr1[1].Download();
    // prev1[0].Download();
    // prev1[1].Download();

    int t2 = program.Dispatch1D("convolve1DC", num_of_samples, GRP, *prev1[0], *curr1[0], *filter1, *out1[0], num_of_samples, filter_size, num_of_channels, samples_per_channel_padd);
    int t3 = program.Dispatch1D("convolve1DC", num_of_samples, GRP, *prev1[1], *curr1[1], *filter1, *out1[1], num_of_samples, filter_size, num_of_channels, samples_per_channel_padd);
    el.Stamp("convolve1D");
    out1[0].Download();
    out1[1].Download();

    // for (int i = 0; i < num_of_channels; ++i)
    // {
    //     FFT::Dispatch(true, curr1[0], i * samples_per_channel_padd, samples_per_channel_padd);
    //     FFT::Dispatch(true, curr1[1], i * samples_per_channel_padd, samples_per_channel_padd);
    //     // int t4 = program.Dispatch1D("convolve2DFreq", samples_per_channel_padd, GRP, curr1[0].Sub(i * samples_per_channel_padd, samples_per_channel_padd), filterFFT.Sub(i * samples_per_channel_padd, samples_per_channel_padd), samples_per_channel_padd);
    //     // int t5 = program.Dispatch1D("convolve2DFreq", samples_per_channel_padd, GRP, curr1[1].Sub(i * samples_per_channel_padd, samples_per_channel_padd), filterFFT.Sub(i * samples_per_channel_padd, samples_per_channel_padd), samples_per_channel_padd);
    // }
    // curr1[0].Download();
    // curr1[1].Download();
    // int t4 = program.Dispatch1D("convolve2DFreq", num_of_samples_padd, GRP, *curr1[0], *filterFFT, num_of_samples_padd);
    // int t5 = program.Dispatch1D("convolve2DFreq", num_of_samples_padd, GRP, *curr1[1], *filterFFT, num_of_samples_padd);
    // curr1[0].Download();
    // curr1[1].Download();
    // for (int i = 0; i < num_of_channels; ++i)
    // {
    //     // FFT::Dispatch(false, curr1[0], out1[0], i * samples_per_channel_padd, samples_per_channel_padd);
    //     // FFT::Dispatch(false, curr1[1], out1[1], i * samples_per_channel_padd, samples_per_channel_padd);
    //     FFT::Dispatch(false, curr1[0], i * samples_per_channel_padd, samples_per_channel_padd);
    //     FFT::Dispatch(false, curr1[1], i * samples_per_channel_padd, samples_per_channel_padd);
    // }
    // // out1[0].Download();
    // // out1[1].Download();
    // curr1[0].Download();
    // curr1[1].Download();
    int t50 = program.Dispatch1D("InverseTranspose2DPalanarC", num_of_samples, GRP, *out1[0], *out1[1], *out, samples_per_channel, num_of_channels, samples_per_channel_padd);
    // int t50 = program.Dispatch1D("InverseTranspose2DPalanarC", num_of_samples, GRP, *curr1[0], *curr1[1], *out, samples_per_channel, num_of_channels, samples_per_channel_padd);

    // int t0 = FFT::Dispatch(true, *curr, *out, num_of_samples);
    std::cout << "Transpose: " << t0 << "us, " << t1 << "us\n";
    // std::cout << "Convolve: " << t2 << "us, " << t3 << "us\n";
    // std::cout << "InvTranspose: " << t4 << "us\n";
    el.Stamp("FFT");

    out.Download();
    el.Stamp("Data downloaded");

    int errors = Compare(&out, &result, num_of_samples);
    el.Stamp("Compare");

    std::cout << "Errors: " << errors << std::endl;

    std::cout << "----------------------------------------------------------\n";
    return 0;
}
