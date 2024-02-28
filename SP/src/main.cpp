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
const int samples_per_channel_padd = FFT::NextPow2(samples_per_channel + filter_size * 2 - 1);
const int num_of_samples = num_of_channels * samples_per_channel;
const int num_of_filters = num_of_channels * filter_size;
const int num_of_samples_padd = num_of_channels * samples_per_channel_padd;

int Compare(const std::vector<std::complex<float>> &a, const std::vector<std::complex<float>> &b, size_t size = 0)
{
    if (size <= 0)
        size = a.size();
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
int Compare(const std::vector<float> &a, const std::vector<float> &b, size_t size = 0)
{
    if (size <= 0)
        size = a.size();

    int errors = 0;
    for (auto i = 0U; i < size; ++i)
    {
        if (std::abs(a[i] - b[i]) > 1e-6)
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
    el.Stamp("FFT init");
    Program program("../src/kernels/main.cl");

    // files for testing
    Buffer<int> gl(10);
    Buffer<float> filter, abs_out(num_of_samples), abs_prev, fir_out;
    Buffer<std::complex<float>> fft_out;
    Buffer<std::complex<float>> prev, curr, out(num_of_samples), result;

    // gpu for algorithms
    Buffer<float> firFilter, currAbs(num_of_samples), currFir(num_of_samples);
    Buffer<std::complex<float>> filterFFT(num_of_samples_padd);
    Buffer<std::complex<float>> prevT(num_of_samples_padd), currT(num_of_samples_padd);
    const float F7 = 1.f / 7.f;
    firFilter.host().resize(7, F7);

    //    filter.ReadFromFile("../../data/4/filter.bin");
    filter.ReadFromFile("../../data/FILTER.32fc");
    curr.ReadFromFile("../../data/Chann_current4.32fc");
    prev.ReadFromFile("../../data/Chann_prev4.32fc");
    result.ReadFromFile("../../data/Chann_out4.32fc", false);
    fft_out.ReadFromFile("../../data/FFT_OUT4.32fc");
    abs_prev.ReadFromFile("../../data/ABS_OUT3.32fc"); //
    abs_out.ReadFromFile("../../data/ABS_OUT4.32fc");
    fir_out.ReadFromFile("../../data/FIR_OUT3.32fc");

    if (filter.size() < num_of_filters || curr.size() != num_of_samples || prev.size() != num_of_samples || result.size() != num_of_samples)
    {
        std::cerr << "Invalid data size" << std::endl;
        return 1;
    }
    el.Stamp("Data loaded");

    gl.Refresh();
    prevT.Refresh();
    currT.Refresh();
    filterFFT.Refresh();
    currAbs.Refresh();
    firFilter.Refresh();
    currFir.Refresh();
    el.Stamp("Data allocated");

    program.Dispatch1D("Transpose1DC", num_of_filters, GRP, *filter, *filterFFT, num_of_channels, filter_size, samples_per_channel_padd, 1);
    int tot = 0;
    int errors = 0;
    // FFT::Dispatch(true, filterFFT, num_of_channels);
    for (int i = 0; i < num_of_channels; ++i)
        tot += FFT::Dispatch(true, filterFFT, i * samples_per_channel_padd, samples_per_channel_padd);
    el.Stamp("FFT filter");
    //    for (int i = 0; i < 1000; ++i)
    {
        el.GStamp("Transpose Signal", program.Dispatch1D("Transpose2D", num_of_samples, GRP, *curr, *currT, num_of_channels, samples_per_channel, samples_per_channel_padd, filter_size));
        el.GStamp("Transpose copypv", program.Dispatch1D("Transpose2D_copy_Prev", filter_size * samples_per_channel, GRP, *prev, *currT, num_of_channels, samples_per_channel, filter_size, samples_per_channel_padd));
        el.Stamp("Transpose Signal");

        el.GStamp("FWD FFT", FFT::Dispatch(true, currT, num_of_channels));
        el.GStamp("convolve2DFreq", program.Dispatch1D("convolve2DFreq", num_of_samples_padd, GRP, *currT, *filterFFT, num_of_samples_padd));
        el.GStamp("BK FFT", FFT::Dispatch(false, currT, num_of_channels));
        el.Stamp("FFT conv");

        el.GStamp("InverseTranspose Result2", program.Dispatch1D("InverseTranspose2D", num_of_samples, GRP, *currT, *out, samples_per_channel, num_of_channels, samples_per_channel_padd, filter_size));
        el.GStamp("FFT V", FFT::Dispatch(true, out, samples_per_channel));
        el.GStamp("ABS", program.Dispatch1D("AbsMag", num_of_samples, GRP, *out, *currAbs, num_of_samples));
        el.GStamp("convolve1DFir", program.Dispatch1D("convolve1DFir", num_of_samples, GRP, *abs_prev, *currAbs, *currFir, samples_per_channel, num_of_channels));
    }
    currAbs.Download();
    currFir.Download();
    el.Stamp("Data downloaded");

    errors = Compare(&currAbs, &abs_out, num_of_samples);
    std::cout << "Errors: " << errors << std::endl;
    errors = Compare(&currFir, &fir_out, num_of_samples);
    std::cout << "Errors: " << errors << std::endl;
    el.Stamp("Compare");
    std::cout << "----------------------------------------------------------\n";

    return 0;
}
