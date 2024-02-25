#include "Buffer.hpp"
#include <clFFT.h>

class FFT
{
private:
    struct Plan
    {
        Plan(const Plan &) = default;
        Plan(size_t size, clfftLayout layout = CLFFT_COMPLEX_INTERLEAVED, clfftResultLocation _placeness = CLFFT_OUTOFPLACE, clfftPrecision _precision = CLFFT_SINGLE);
        Plan(size_t sizex, size_t sizey, clfftLayout layout = CLFFT_COMPLEX_INTERLEAVED, clfftResultLocation _placeness = CLFFT_OUTOFPLACE, clfftPrecision _precision = CLFFT_SINGLE);
        Plan(size_t sizex, size_t sizey, size_t sizez, clfftLayout layout = CLFFT_COMPLEX_INTERLEAVED, clfftResultLocation _placeness = CLFFT_OUTOFPLACE, clfftPrecision _precision = CLFFT_SINGLE);
        ~Plan();

        bool Init();
        std::string Key() const;
        clfftPlanHandle &operator()() { return handle; }

        clfftPlanHandle handle = 0;
        clfftDim dims = CLFFT_1D;
        size_t sizes[CLFFT_3D] = {0, 0, 0};
        size_t batchSize = 1;
        size_t dist = 0;
        clfftPrecision precision = CLFFT_SINGLE;
        clfftResultLocation placeness = CLFFT_OUTOFPLACE;
        clfftLayout iLayout = CLFFT_COMPLEX_INTERLEAVED;
        clfftLayout oLayout = CLFFT_COMPLEX_INTERLEAVED;
        size_t workSize = 0;
    };
    using PlanPtr = std::shared_ptr<Plan>;

private:
    FFT();
    ~FFT();

public:
    static FFT &getInstance();
    static size_t NextPow2(size_t n);
    static size_t NextPow235(size_t n);
    static int Dispatch(bool fwd, cl::Buffer &inputBuffer, cl::Buffer &outputBuffer, size_t size, size_t split, size_t dist = 0);
    template <typename T>
    static int Dispatch(bool fwd, Buffer<T> &inputBuffer, size_t split)
    {
        return Dispatch(fwd, *inputBuffer, null, inputBuffer.size(), split);
    }
    template <typename T>
    static int Dispatch(bool fwd, Buffer<T> &inputBuffer, Buffer<T> &outputBuffer, size_t split)
    {
        return Dispatch(fwd, *inputBuffer, *outputBuffer, inputBuffer.size(), split);
    }
    template <typename T>
    static int Dispatch(bool fwd, Buffer<T> &buffer, size_t offset, size_t size)
    {
        return Dispatch(fwd, buffer.Sub(offset, size), null, size, 1);
    }
    template <typename T>
    static int Dispatch(bool fwd, Buffer<T> &inputBuffer, Buffer<T> &outputBuffer, size_t offset, size_t size)
    {
        return Dispatch(fwd, inputBuffer.Sub(offset, size), outputBuffer.Sub(offset, size), size, 1);
    }

private:
    int dispatch(const Plan &plan, bool fwd, cl::Buffer &inputBuffer, cl::Buffer &outputBuffer);
    void AddPlan(PlanPtr plan);

private:
    cl::Buffer tmpBuffer;
    size_t maxWorkSize = 0;
    std::map<std::string, PlanPtr> plans;
    static cl::Buffer null;
};
