#pragma once

#include "GPU.h"

NAMESPACE_BEGIN(cu);

using u8 = unsigned char;

class BMP
{
public:
    static bool SignalReal2BMP(const std::string &filename, const float *data, int width, int height);
    static bool SignalComplex2BMP(const std::string &filename, const float2 *data, int width, int height);
    static bool STFTComplex2BMP(const std::string &filename, const float2 *data, int width, int height);

private:
    static bool RGBA2BMP(const std::string &filename, const u8 *data, int width, int height);
};

NAMESPACE_END(cu);