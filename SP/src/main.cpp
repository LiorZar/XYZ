#include "cl/defines.h"
#include "cl/Program.h"
#include "cl/Buffer.hpp"
#include "cl/FFT.h"
#include "cl/Elapse.hpp"

const int GRP = 256;
const int num_of_channels = 20;
const int samples_per_channel = 20000;
const int filter_size = 500;
const int samples_per_channel_padd = 20499;
const int num_of_samples = num_of_channels * samples_per_channel;
const int num_of_filters = num_of_channels * filter_size;
const int num_of_samples_padd = num_of_channels * samples_per_channel_padd;

int Compare(const std::vector<std::complex<float>> &a, const std::vector<std::complex<float>> &b)
{
    int errors = 0;
    for (auto i = 0U; i < a.size(); ++i)
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

int main()
{

    Elapse el("Main");
    std::cout << "----------------------------------------------------------\n";
    Context::getInstance();
    FFT::getInstance();
    Program program("../../src/kernels/main.cl");

    Buffer<float> filter, filter1(num_of_filters);
    Buffer<std::complex<float>> curr1[2], prev1[2], out1[2];
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
        curr1[i].Resize(num_of_samples);
        prev1[i].Resize(num_of_samples);
        out1[i].Resize(num_of_samples);
    }
    // int tt = program.Dispatch1D("convolve2D", num_of_samples, GRP, *prev, *curr, *filter, *out, num_of_samples, filter_size, num_of_channels);
    // int t0 = program.Dispatch1D("Transpose2DPalanar", num_of_samples, GRP, *curr, *currx, *curry, num_of_channels, samples_per_channel);
    int tt = program.Dispatch1D("Transpose1D", num_of_filters, GRP, *filter, *filter1, num_of_channels, filter_size, filter_size);
    int t0 = program.Dispatch1D("Transpose2DPalanarC", num_of_samples, GRP, *curr, *curr1[0], *curr1[1], num_of_channels, samples_per_channel, samples_per_channel);
    int t1 = program.Dispatch1D("Transpose2DPalanarC", num_of_samples, GRP, *prev, *prev1[0], *prev1[1], num_of_channels, samples_per_channel, samples_per_channel);
    el.Stamp("Transpose2DPalanar");
    filter1.Download();
    curr1[0].Download();
    curr1[1].Download();
    prev1[0].Download();
    prev1[1].Download();

    int t2 = program.Dispatch1D("convolve1DC", num_of_samples, GRP, *prev1[0], *curr1[0], *filter1, *out1[0], num_of_samples, filter_size, num_of_channels);
    int t3 = program.Dispatch1D("convolve1DC", num_of_samples, GRP, *prev1[1], *curr1[1], *filter1, *out1[1], num_of_samples, filter_size, num_of_channels);
    el.Stamp("convolve1D");
    out1[0].Download();
    out1[1].Download();

    int t4 = program.Dispatch1D("InverseTranspose2DPalanarC", num_of_samples, GRP, *out1[0], *out1[1], *out, samples_per_channel, num_of_channels, num_of_channels);

    // int t0 = FFT::Dispatch(true, *curr, *out, num_of_samples);
    std::cout << "Transpose: " << t0 << "us, " << t1 << "us\n";
    std::cout << "Convolve: " << t2 << "us, " << t3 << "us\n";
    std::cout << "InvTranspose: " << t4 << "us\n";
    el.Stamp("FFT");

    out.Download();
    el.Stamp("Data downloaded");

    int errors = Compare(&out, &result);
    el.Stamp("Compare");

    std::cout << "Errors: " << errors << std::endl;

    std::cout << "----------------------------------------------------------\n";
    return 0;
}
