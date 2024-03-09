#ifndef __BMP_H__
#define __BMP_H__

#include "GPU.h"

NAMESPACE_BEGIN(cu);

using u8 = unsigned char;
enum eType
{
    Real = 0,
    Imaginary = 1,
    Magnitute = 2
};

class BMP
{
private:
    BMP();
    BMP(const BMP &);

public:
    static BMP *Get();
    ~BMP();

public:
    static bool SignalReal2BMP(const std::string &filename, const gbuffer<float> &data, int width, int height, int line_width = 1);
    static bool SignalComplex2BMP(const std::string &filename, const gbuffer<float2> &data, int width, int height, eType type, int line_width = 1);
    static bool STFTComplex2BMP(const std::string &filename, const gbuffer<float2> &data, int dataWidth, int dataHeight, int bmpWidth, int bmpHeight, eType type);

private:
    static bool Normalize(const gbuffer<float> &data, int width);
    static bool RGBA2BMP(const std::string &filename, const u8 *data, int width, int height);
};

NAMESPACE_END(cu);

#endif //__BMP_H__