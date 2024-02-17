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
    cl::Buffer tmpBuffer;
};
