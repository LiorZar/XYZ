#include "Context.h"
#include <clFFT.h>

class FFT
{
    FFT();
    ~FFT();

public:
    static FFT &getInstance();
    static int Dispatch(bool fwd, cl::Buffer &inputBuffer, cl::Buffer &outputBuffer, size_t size);

private:
    int dispatch(bool fwd, cl::Buffer &inputBuffer, cl::Buffer &outputBuffer, size_t size);
    void AllocateTmpBuffer(const clfftPlanHandle &plan);

private:
    struct Plan
    {
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
        clfftPrecision precision = CLFFT_SINGLE;
        clfftResultLocation placeness = CLFFT_OUTOFPLACE;
        clfftLayout iLayout = CLFFT_COMPLEX_INTERLEAVED;
        clfftLayout oLayout = CLFFT_COMPLEX_INTERLEAVED;

        cl::Buffer tmpBuffer;
    };
    using PlanPtr = std::shared_ptr<Plan>;
    std::map<std::string, PlanPtr> plans;

private:
    cl::Buffer tmpBuffer;
};
