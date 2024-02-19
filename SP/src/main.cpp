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
    el.Stamp("FFT initialized");

    Program program("../../src/kernels/main.cl");

    Buffer<float> filter;
    Buffer<std::complex<float>> filterFFT(num_of_samples_padd);
    Buffer<std::complex<float>> curr, prev, out(num_of_samples), result;
    Buffer<std::complex<float>> currT(num_of_samples_padd), prevT(num_of_samples_padd), outT(num_of_samples_padd);

    int tot;
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
    currT.Refresh();
    prevT.Refresh();
    outT.Refresh();
    out.Refresh();
    el.Stamp("Data allocated");

    program.Dispatch1D("Transpose1DC", num_of_filters, GRP, *filter, *filterFFT, num_of_channels, filter_size, samples_per_channel_padd, 1);
    for (int i = 0; i < num_of_channels; ++i)
        FFT::Dispatch(true, filterFFT, i * samples_per_channel_padd, samples_per_channel_padd);
    el.Stamp("FFT filter");

    el.GStamp("Transpose Signal", program.Dispatch1D("Transpose2D", num_of_samples, GRP, *curr, *currT, num_of_channels, samples_per_channel, samples_per_channel_padd));
    el.Stamp("Transpose Signal");

    tot = 0;
    for (int i = 0; i < num_of_channels; ++i)
        tot += FFT::Dispatch(true, currT, i * samples_per_channel_padd, samples_per_channel_padd);
    el.GStamp("FFT curr", tot);
    el.GStamp("convolve2DFreq", program.Dispatch1D("convolve2DFreq", num_of_samples_padd, GRP, *currT, *filterFFT, num_of_samples_padd));

    tot = 0;
    for (int i = 0; i < num_of_channels; ++i)
        tot += FFT::Dispatch(false, currT, i * samples_per_channel_padd, samples_per_channel_padd);
    el.GStamp("InvFFT curr", tot);
    el.GStamp("Inverse Transpose", program.Dispatch1D("InverseTranspose2D", num_of_samples, GRP, *currT, *out, num_of_channels, samples_per_channel, samples_per_channel_padd));
    el.Stamp("FFT");

    out.Download();
    el.Stamp("Data downloaded");

    int errors = Compare(&out, &result, num_of_samples);
    el.Stamp("Compare");

    std::cout << "Errors: " << errors << std::endl;
    std::cout << "----------------------------------------------------------\n";
    return 0;
}
