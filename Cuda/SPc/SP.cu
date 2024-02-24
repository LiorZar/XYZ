#include "SP.cuh"

NAMESPACE_BEGIN(cu);

//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void addVectors(const float *a, const float *b, float *c, const int size);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
const bool useHost = true;
//-------------------------------------------------------------------------------------------------------------------------------------------------//
SP::SP() : firFilter(0, useHost), currAbs(num_of_samples, useHost), currFir(num_of_samples, useHost),
           filterFFT(num_of_samples_padd, useHost),
           prevT(num_of_samples_padd, useHost), currT(num_of_samples_padd, useHost),

           filter(0, useHost), abs_out(num_of_samples, useHost), abs_prev(0, useHost), fir_out(0, useHost),
           fft_out(0, useHost),
           prev(0, useHost), curr(0, useHost), out(num_of_samples, useHost), result(0, useHost)
{
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool SP::Init()
{
    const auto &workDir = GPU::GetWorkingDirectory() + "../../../data/";
    const float F7 = 1.f / 7.f;
    firFilter.resize(7, F7);

    //    filter.ReadFromFile("../../data/4/filter.bin");
    filter.ReadFromFile(workDir + "FILTER.32fc");
    curr.ReadFromFile(workDir + "Chann_current2.32fc");
    prev.ReadFromFile(workDir + "Chann_prev2.32fc");
    result.ReadFromFile(workDir + "Chann_out2.32fc", false);
    fft_out.ReadFromFile(workDir + "FFT_OUT2.32fc");
    abs_prev.ReadFromFile(workDir + "FIR_PREV2.32fc"); //
    abs_out.ReadFromFile(workDir + "ABS_OUT2.32fc");
    fir_out.ReadFromFile(workDir + "FIR_OUT2.32fc");

    if (filter.size() < num_of_filters || curr.size() != num_of_samples || prev.size() != num_of_samples || result.size() != num_of_samples)
    {
        std::cerr << "Invalid data size" << std::endl;
        return false;
    }

    addVectors<<<1, 1000>>>(*filter, *abs_prev, *abs_out, 1000);
    cu::sync();

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void addVectors(const float *a, const float *b, float *c, const int size)
{
    const int I = getI();
    if (I >= size)
        return;

    c[I] = a[I] + b[I];
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//

NAMESPACE_END(cu);