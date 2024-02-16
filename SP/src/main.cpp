#include "cl/defines.h"
#include "cl/Program.h"
#include "cl/Buffer.hpp"
#include <complex>

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
    std::cout << "----------------------------------------------------------\n";
    Context::getInstance();
    Program program("../../src/kernels/main.cl");

    Buffer<float> filter;
    Buffer<std::complex<float>> curr, prev, out(num_of_samples), result;

    curr.ReadFromFile("../../../data/0/curr.bin");
    prev.ReadFromFile("../../../data/0/prev.bin");
    filter.ReadFromFile("../../../data/0/filter.bin");
    result.ReadFromFile("../../../data/0/result.bin", false);
    if (filter.size() != num_of_filters || curr.size() != num_of_samples || prev.size() != num_of_samples || result.size() != num_of_samples)
    {
        std::cerr << "Invalid data size" << std::endl;
        return 1;
    }

    int t0 = program.Dispatch1D("convolve", num_of_samples, 256, *prev, *curr, *filter, *out, num_of_samples, filter_size, num_of_channels);
    std::cout << "Time: " << t0 << "ns\n";
    out.Download();
    int errors = Compare(&out, &result);
    std::cout << "Errors: " << errors << std::endl;

    std::cout << "----------------------------------------------------------\n";
    return 0;
}
