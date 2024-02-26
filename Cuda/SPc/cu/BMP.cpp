#include "BMP.h"

NAMESPACE_BEGIN(cu);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
inline float magnitude(float2 a)
{
    return std::sqrt(a.x * a.x + a.y * a.y);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
inline float lerp(float a, float b, float t)
{
    return a + t * (b - a);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
inline float invLerp(float a, float b, float v)
{
    return (v - a) / (b - a);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
inline int toY(float v, float min, float max, int height)
{
    return static_cast<int>(lerp(0.f, (float)height, invLerp(min, max, v)));
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
inline void toColor(float v, u8 *rgb)
{
    static const u8 RAINBOW[7][3] = {
        {0, 0, 255},
        {0, 255, 255},
        {0, 255, 0},
        {255, 255, 0},
        {255, 0, 0},
        {255, 0, 255},
        {255, 0, 255}};

    int i = static_cast<int>(v * 6);
    float f = (v * 6) - i;
    rgb[0] = static_cast<u8>(lerp(RAINBOW[i][0], RAINBOW[i + 1][0], f));
    rgb[1] = static_cast<u8>(lerp(RAINBOW[i][1], RAINBOW[i + 1][1], f));
    rgb[2] = static_cast<u8>(lerp(RAINBOW[i][2], RAINBOW[i + 1][2], f));
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void MinMax(const float *data, int size, float &min, float &max)
{
    min = max = data[0];
    for (int i = 1; i < size; i++)
    {
        if (data[i] < min)
            min = data[i];
        if (data[i] > max)
            max = data[i];
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Complex2Mag(const float2 *data, float *mag, int size)
{
    for (int i = 0; i < size; i++)
        mag[i] = magnitude(data[i]);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool BMP::SignalReal2BMP(const std::string &filename, const float *data, int width, int height, int line_width)
{
    float minVal, maxVal;
    MinMax(data, width, minVal, maxVal);

    std::vector<u8> buffer(width * height * 4, 0);
    for (int x = 0; x < width; x++)
    {
        int Y = toY(data[x], minVal, maxVal, height);
        for (int y = Y - line_width; y < Y + line_width; ++y)
        {
            if (y < 0 || y >= height)
                continue;

            int i = x + y * width;
            buffer[i * 4 + 0] = buffer[i * 4 + 1] = buffer[i * 4 + 2] = 255;
            buffer[i * 4 + 3] = 255;
        }
    }

    return RGBA2BMP(filename, buffer.data(), width, height);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool BMP::SignalComplex2BMP(const std::string &filename, const float2 *cdata, int width, int height, int line_width)
{
    std::vector<float> data(width);
    Complex2Mag(cdata, data.data(), width);
    return SignalReal2BMP(filename, data.data(), width, height, line_width);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool BMP::STFTComplex2BMP(const std::string &filename, const float2 *cdata, int width, int height)
{
    std::vector<float> data(width * height);
    Complex2Mag(cdata, data.data(), width * height);
    float minVal, maxVal;
    MinMax(data.data(), width * height, minVal, maxVal);

    std::vector<u8> buffer(width * height * 4);
    for (int i = 0; i < width * height; i++)
        toColor(invLerp(minVal, maxVal, data[i]), buffer.data() + i * 4);

    return RGBA2BMP(filename, buffer.data(), width, height);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool BMP::RGBA2BMP(const std::string &filename, const unsigned char *data, int width, int height)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
        return false;

    int size = 54 + 4 * width * height;
    u8 header[54] = {0};
    header[0] = 'B';
    header[1] = 'M';
    header[2] = size;
    header[3] = size >> 8;
    header[4] = size >> 16;
    header[5] = size >> 24;
    header[10] = 54;
    header[14] = 40;
    header[18] = width;
    header[19] = width >> 8;
    header[20] = width >> 16;
    header[21] = width >> 24;
    header[22] = height;
    header[23] = height >> 8;
    header[24] = height >> 16;
    header[25] = height >> 24;
    header[26] = 1;
    header[28] = 32;

    file.write(reinterpret_cast<const char *>(header), 54);
    file.write(reinterpret_cast<const char *>(data), 4 * width * height);
    file.close();

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(cu);