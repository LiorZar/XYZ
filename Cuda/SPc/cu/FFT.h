#pragma once

#include "GPU.h"

NAMESPACE_BEGIN(cu);

class FFT
{
private:
    FFT();
    ~FFT();

public:
    static FFT &getInstance();
    static size_t NextPow2(size_t n);
    static size_t NextPow235(size_t n);
};

NAMESPACE_END(cu);