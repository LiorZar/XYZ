#include "cl/defines.h"
#include "cl/Program.h"
#include "cl/Buffer.hpp"
#include "cl/FFT.h"
#include "cl/Elapse.hpp"

#include <complex>
#include <cmath>

const int num_of_channels = 20;
const int samples_per_channel = 20000;
const int filter_size = 500;
const int num_of_samples = num_of_channels * samples_per_channel;
const int num_of_filters = num_of_channels * filter_size;

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

    Buffer<float> filter;
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

    // int t0 = program.Dispatch1D("convolve", num_of_samples, 256, *prev, *curr, *filter, *out, num_of_samples, filter_size, num_of_channels);
    int t0 = FFT::Dispatch(true, *curr, *out, num_of_samples);
    std::cout << "Time: " << t0 << "us\n";
    el.Stamp("FFT");

    out.Download();
    el.Stamp("Data downloaded");

    int errors = Compare(&out, &result);
    el.Stamp("Compare");

    std::cout << "Errors: " << errors << std::endl;

    std::cout << "----------------------------------------------------------\n";
    return 0;
}
