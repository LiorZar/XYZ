#include "FFT.h"
#include <exception>

//-------------------------------------------------------------------------------------------------------------------------------------------------//
FFT::FFT()
{
    clfftSetupData fftSetup;
    auto err0 = clfftInitSetupData(&fftSetup);
    auto err1 = clfftSetup(&fftSetup);

    std::cout << "clfftInitSetupData: " << err0 << std::endl;
    std::cout << "clfftSetup: " << err1 << std::endl;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
FFT::~FFT()
{
    clfftTeardown();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
FFT &FFT::getInstance()
{
    static FFT instance;
    return instance;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
int FFT::Dispatch(bool fwd, cl::Buffer &inputBuffer, cl::Buffer &outputBuffer, size_t size)
{
    return getInstance().dispatch(fwd, inputBuffer, outputBuffer, size);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
int FFT::dispatch(bool fwd, cl::Buffer &inputBuffer, cl::Buffer &outputBuffer, size_t size)
{
    try
    {
        auto &queue = Context::Q();

        clfftPlanHandle plan;
        clfftCreateDefaultPlan(&plan, Context::get()(), CLFFT_1D, &size);

        clfftSetPlanBatchSize(plan, 1);
        clfftSetPlanPrecision(plan, CLFFT_SINGLE);
        clfftSetResultLocation(plan, CLFFT_OUTOFPLACE);
        clfftSetLayout(plan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);

        clfftBakePlan(plan, 1, &queue(), nullptr, nullptr);
        AllocateTmpBuffer(plan);

        // Enqueue the FFT operation
        cl::Event event;
        auto status = clfftEnqueueTransform(plan, fwd ? CLFFT_FORWARD : CLFFT_BACKWARD, 1, &queue(), 0, NULL, &event(), &inputBuffer(), &outputBuffer(), nullptr);

#ifdef TIMING
        queue.finish();
        event.wait();
        auto e = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        auto s = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        // clfftDestroyPlan(&plan);

        return int((e - s) / 1000);
#endif
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }

    return 0;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void FFT::AllocateTmpBuffer(const clfftPlanHandle &plan)
{
    size_t tmpBufferSize;
    clfftGetTmpBufSize(plan, &tmpBufferSize);

    if (tmpBuffer() == nullptr || tmpBuffer.getInfo<CL_MEM_SIZE>() < tmpBufferSize)
        tmpBuffer = cl::Buffer(Context::get(), CL_MEM_READ_WRITE, tmpBufferSize);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
