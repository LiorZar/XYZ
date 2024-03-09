#include "BMP.cuh"

NAMESPACE_BEGIN(cu);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
const int FRESL = 1000000;
const float IFRESEL = 1.f / float(FRESL);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
const bool useHost = false;
//-------------------------------------------------------------------------------------------------------------------------------------------------//
cudaTextureDesc texDesc;
cudaResourceDesc resDesc;
cudaChannelFormatDesc chanDesc = cudaCreateChannelDesc<float>();
//-------------------------------------------------------------------------------------------------------------------------------------------------//
gbuffer<int> gl(10, useHost);
gbuffer<u8> image(0, useHost);
gbuffer<float> tmpBuffer(0, useHost);
gbuffer<float> nrmBuffer(0, useHost);
gbuffer<float> imageVals(0, useHost);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__host__ __device__ inline float magnitude(float2 a)
{
    return std::sqrt(a.x * a.x + a.y * a.y);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__host__ __device__ inline void toColor(float v, u8 *rgb)
{
    static const u8 RAINBOW[10][3] = {
        {128, 0, 0},
        {255, 0, 0},
        {255, 128, 0},
        {255, 255, 0},
        {0, 255, 128},
        {0, 255, 255},
        {0, 128, 255},
        {0, 0, 255},
        {0, 0, 128},
        {0, 0, 128}};

    int i = int(v * 8);
    float f = (v * 8) - i;
    rgb[0] = u8(Lerp(RAINBOW[i][0], RAINBOW[i + 1][0], f));
    rgb[1] = u8(Lerp(RAINBOW[i][1], RAINBOW[i + 1][1], f));
    rgb[2] = u8(Lerp(RAINBOW[i][2], RAINBOW[i + 1][2], f));
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
template <typename T>
__global__ void zeroBuffer(T *buffer, int size)
{
    int idx = getI();
    if (idx < size)
        buffer[idx] = T(0);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void drawPolyline(float *output, const float *src, int width, int height, int lineWidth)
{
    int x = getI();
    if (x >= width - 1)
        return;

    float y0 = src[x] * height;
    float y1 = src[x + 1] * height;
    float count = abs(y1 - y0) / 0.25f;
    for (float j = 0; j < count; ++j)
    {
        int y = int(Lerp(y0, y1, j / (count - 1)));
        for (int i = -lineWidth / 2; i <= lineWidth / 2; ++i)
        {
            int k = x + i;
            int idx = y * width + k;
            if (idx >= 0 && idx < width * height)
                output[idx] = 1.0;
        }
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void tex2rgba(u8 *rgba, cudaTextureObject_t tex, int width, int height)
{
    int J = getI();
    int I = getJ();
    if (I >= height || J >= width)
        return;

    const int idx = (I * width + J) * 4;
    float fx = float(J) / float(width - 1);
    float fy = float(I) / float(height - 1);
    float val = tex2D<float>(tex, fx, fy);
    u8 v = u8(val * 255);
    rgba[idx + 0] = v;
    rgba[idx + 1] = v;
    rgba[idx + 2] = v;
    rgba[idx + 3] = 255;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void tex2rainbowT(u8 *rgba, cudaTextureObject_t tex, int width, int height)
{
    int J = getI();
    int I = getJ();
    if (I >= height || J >= width)
        return;

    const int idx = (I * width + J) * 4;
    float fx = float(J) / float(width - 1);
    float fy = float(I) / float(height - 1);
    float val = tex2D<float>(tex, 1.f - fy, fx);
    toColor(val, rgba + idx);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void findBufferMinMax(const float *buffer, int size, int *gl)
{
    int idx = getI();
    if (idx >= size)
        return;

    atomicMin(&gl[0], int(buffer[idx] * FRESL));
    atomicMax(&gl[1], int(buffer[idx] * FRESL));
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void normalizeBuffer(float *output, const float *src, int size, float minVal, float maxVal)
{
    int idx = getI();
    if (idx >= size)
        return;

    float v = src[idx];
    v = InvLerp(minVal, maxVal, v);
    output[idx] = v;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
__global__ void complex2float(float *output, const float2 *src, int size, int type)
{
    int idx = getI();
    if (idx >= size)
        return;

    float v = 0;
    float2 val = src[idx];
    if (0 == type) // real
        v = val.x;
    if (1 == type) // imag
        v = val.y;
    if (2 == type)
        v = magnitude(val);
    output[idx] = v;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
BMP::BMP()
{
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = true;

    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
BMP *BMP::Get()
{
    static BMP instance;
    return &instance;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
BMP::~BMP()
{
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool BMP::SignalReal2BMP(const std::string &filename, const gbuffer<float> &data, int width, int height, int line_width)
{
    Normalize(data, width);
    imageVals.resize(width * height);
    image.resize(width * height * 4);

    drawPolyline<<<DIV(width, BLOCK), BLOCK>>>(*imageVals, *nrmBuffer, width, height, line_width);
    zeroBuffer<<<DIV((int)image.size(), BLOCK), BLOCK>>>(*image, (int)image.size());

    cudaArray *d_imageVals;
    cudaMallocArray(&d_imageVals, &chanDesc, width, height);
    cudaMemcpy2DToArray(d_imageVals, 0, 0, *imageVals, width * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToDevice);
    resDesc.res.array.array = d_imageVals;

    dim3 grid(DIV(width, 16), DIV(height, 16)), block(16, 16);
    cudaTextureObject_t tex;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
    tex2rgba<<<grid, block>>>(*image, tex, width, height);
    cudaDestroyTextureObject(tex);
    cudaFreeArray(d_imageVals);

    image.RefreshDown();
    return RGBA2BMP(filename, image.h(), width, height);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool BMP::SignalComplex2BMP(const std::string &filename, const gbuffer<float2> &cdata, int width, int height, eType type, int line_width)
{
    tmpBuffer.resize(width);
    complex2float<<<GRID(width), BLOCK>>>(*tmpBuffer, *cdata, width, (int)type);
    return SignalReal2BMP(filename, tmpBuffer, width, height, line_width);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool BMP::STFTComplex2BMP(const std::string &filename, const gbuffer<float2> &cdata, int dataWidth, int dataHeight, int bmpWidth, int bmpHeight, eType type)
{
    const int dataSize = dataWidth * dataHeight;
    tmpBuffer.resize(dataSize);
    complex2float<<<GRID(dataSize), BLOCK>>>(*tmpBuffer, *cdata, dataSize, (int)type);
    Normalize(tmpBuffer, dataSize);

    image.resize(bmpWidth * bmpHeight * 4);

    cudaArray *d_imageVals;
    cudaMallocArray(&d_imageVals, &chanDesc, dataWidth, dataHeight);
    cudaMemcpy2DToArray(d_imageVals, 0, 0, *nrmBuffer, dataWidth * sizeof(float), dataWidth * sizeof(float), dataHeight, cudaMemcpyDeviceToDevice);
    resDesc.res.array.array = d_imageVals;

    dim3 grid(DIV(bmpWidth, 16), DIV(bmpHeight, 16)), block(16, 16);
    cudaTextureObject_t tex;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
    tex2rainbowT<<<grid, block>>>(*image, tex, bmpWidth, bmpHeight);
    cudaDestroyTextureObject(tex);
    cudaFreeArray(d_imageVals);

    image.RefreshDown();
    return RGBA2BMP(filename, image.h(), bmpWidth, bmpHeight);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool BMP::Normalize(const gbuffer<float> &data, int width)
{
    gl[0] = FRESL;
    gl[1] = -FRESL;
    gl.RefreshUp();
    findBufferMinMax<<<GRID(width), BLOCK>>>(*data, width, *gl);
    gl.RefreshDown();

    float minVal = gl[0] * IFRESEL, maxVal = gl[1] * IFRESEL;
    float dVal = maxVal - minVal;
    if (maxVal - minVal < 0.00001f) // two normalized
    {
        maxVal += 0.5f;
        minVal -= 0.5f;
    }
    else
    {
        minVal -= dVal * 0.01f; // one percent enlarge
        maxVal += dVal * 0.01f; // one percent enlarge
    }
    nrmBuffer.resize(width);
    normalizeBuffer<<<GRID(width), BLOCK>>>(*nrmBuffer, *data, width, minVal, maxVal);

    return true;
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