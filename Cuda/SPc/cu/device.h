#pragma once
//--------------------------------------------------------------------------------------------------------------------//
#include "helper_math.h"
//--------------------------------------------------------------------------------------------------------------------//
__device__ __forceinline__ int getI() { return (blockIdx.x * blockDim.x + threadIdx.x); }
__device__ __forceinline__ int getJ() { return (blockIdx.y * blockDim.y + threadIdx.y); }
__device__ __forceinline__ int getK() { return (blockIdx.z * blockDim.z + threadIdx.z); }
__device__ __forceinline__ int2 getIJ() { return make_int2(getI(), getJ()); }
__device__ __forceinline__ int3 getIJK() { return make_int3(getI(), getJ(), getK()); }
//--------------------------------------------------------------------------------------------------------------------//
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif
__device__ const float SQRT2 = 1.4142135623730950488016887242097f;
__device__ const float ISQRT2 = 0.70710678118654752440084436210485f;
__device__ const float SQRT3 = 1.7320508075688772935274463415059f;
__device__ const float ISQRT3 = 0.57735026918962576450914878050196f;
//--------------------------------------------------------------------------------------------------------------------//
__device__ const float EPS = 0.00001f;
__device__ const float EMPTY_VALUE = -10000.f;
__device__ const float MAX_DISTANCE = 1000.f;
//--------------------------------------------------------------------------------------------------------------------//
__device__ const int3 ngb[] =
    {
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}, {-1, 0, -1}, {1, 0, -1}, {-1, 0, 1}, {1, 0, 1}, {-1, -1, 0}, {1, -1, 0}, {-1, 1, 0}, {1, 1, 0}, {0, -1, -1}, {0, -1, 1}, {0, 1, 1}, {0, 1, -1}, {-1, -1, -1}, {1, -1, -1}, {-1, -1, 1}, {1, -1, 1}, {-1, 1, -1}, {1, 1, -1}, {-1, 1, 1}, {1, 1, 1}, {0, 0, 0}};
//--------------------------------------------------------------------------------------------------------------------//
__device__ const int3 ingb[] =
    {
        {1, 0, 0}, {2, 0, 0}, {0, 3, 0}, {0, 4, 0}, {0, 0, 5}, {0, 0, 6}, {1, 0, 5}, {2, 0, 5}, {1, 0, 6}, {2, 0, 6}, {1, 3, 0}, {2, 3, 0}, {1, 4, 0}, {2, 4, 0}, {0, 3, 5}, {0, 3, 6}, {0, 4, 6}, {0, 4, 5}, {1, 3, 5}, {2, 3, 5}, {1, 3, 6}, {2, 3, 6}, {1, 4, 5}, {2, 4, 5}, {1, 4, 6}, {2, 4, 6}, {0, 0, 0}};
//--------------------------------------------------------------------------------------------------------------------//
#define D1 0.9016f
#define D2 1.289f
#define D3 1.615f
//--------------------------------------------------------------------------------------------------------------------//
__device__ const float ngbDists[] =
    {
        D1, D1, D1, D1, D1, D1,
        D2, D2,
        D2, D2,
        D2, D2,
        D3, D3, D2,
        D3, D3, D2,
        D2, D2,
        D3, D3, D2,
        D3, D3, D2};
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ float toResL(float _v, float _r)
{
    return floorf(_v / _r) * _r;
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ float toResH(float _v, float _r)
{
    return ceilf(_v / _r) * _r;
}
//--------------------------------------------------------------------------------------------------------------------//
inline __device__ int ngbIndex(int _baseIdx, int _26Index, const int *_vngb7)
{
    // _ngbId < 26
    int index = _baseIdx;
    int3 shifts = ingb[_26Index];

    index = (shifts.x > 0) ? _vngb7[abs(index) * 7 + shifts.x] : index;
    index = (shifts.y > 0) ? _vngb7[abs(index) * 7 + shifts.y] : index;
    index = (shifts.z > 0) ? _vngb7[abs(index) * 7 + shifts.z] : index;

    return index;
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ int voxVec2voxId(const int3 &v, const int3 &dims)
{
    if (v.x < 0 || v.x >= dims.x)
        return -1;
    if (v.y < 0 || v.y >= dims.y)
        return -1;
    if (v.z < 0 || v.z >= dims.z)
        return -1;

    return v.x + dims.x * (v.y + dims.y * v.z);
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ int3 voxId2voxVec(int voxId, const int3 &dims)
{
    return make_int3(voxId % dims.x, (voxId / dims.x) % dims.y, (voxId / (dims.x * dims.y)) % dims.z);
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ float3 normalizePos(const float3 &pos, const float3 &minBox, const float3 &dBox)
{
    return (pos - minBox) / dBox;
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ float3 pos2Coord(const float3 &pos, const float3 &minBox, float voxelDims)
{
    return (pos - minBox) / voxelDims;
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ int3 pos2voxVec(const float3 &pos, const float3 &minBox, float voxelDims)
{
    return make_int3(pos2Coord(pos, minBox, voxelDims));
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ int pos2voxId(const float3 &pos, const float3 &minBox, float voxelDims, const int3 &dims)
{
    int3 coord = pos2voxVec(pos, minBox, voxelDims);
    return coord.x + dims.x * (coord.y + dims.y * coord.z);
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ float3 voxVec2pos(const int3 &v, const float3 &minBox, float voxelDims)
{
    float3 size = make_float3(v);
    float3 voxBox = make_float3(voxelDims);
    return size * voxBox + minBox + voxBox * 0.5f;
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ float3 voxId2pos(int voxId, const float3 &minBox, float voxelDims, const int3 &dims)
{
    return voxVec2pos(voxId2voxVec(voxId, dims), minBox, voxelDims);
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ bool box2check(const int3 &v, const int3 &minBox, const int3 &maxBox)
{
    if (v.x < minBox.x || v.x >= maxBox.x)
        return false;
    if (v.y < minBox.y || v.y >= maxBox.y)
        return false;
    if (v.z < minBox.z || v.z >= maxBox.z)
        return false;

    return true;
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ int I2I(int I, const int3 &srcDims, const int3 &minBox, const int3 &dstDims)
{
    return voxVec2voxId(minBox + voxId2voxVec(I, srcDims), dstDims);
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ int I2IR(int I, const int3 &srcDims, const int3 &minBox, const int3 &dstDims)
{
    return voxVec2voxId(minBox - make_int3(1, 1, 1) + voxId2voxVec(I, srcDims + make_int3(2, 2, 2)), dstDims);
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ float rangeNormal(float v, float minVal, float maxVal)
{
    return clamp((v - minVal) / (maxVal - minVal), 0.f, 1.f);
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ float mixf(float a, float b, float f)
{
    return a * (1.f - f) + b * f;
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ float2 mixf(const float2 &a, const float2 &b, float f)
{
    return a * (1.f - f) + b * f;
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ float3 mixf(const float3 &a, const float3 &b, float f)
{
    return a * (1.f - f) + b * f;
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ float4 mixf(const float4 &a, const float4 &b, float f)
{
    return a * (1.f - f) + b * f;
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ int3 operator/(const int3 &a, int b)
{
    return make_int3(a.x / b, a.y / b, a.z / b);
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ int3 operator/(const int3 &a, const int3 &b)
{
    return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}
//--------------------------------------------------------------------------------------------------------------------//
__host__ __device__ __inline unsigned int Div(unsigned int v, unsigned int m)
{
    return (v + m - 1) / m;
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ float sqrDistance(const float3 &a, const float3 &b)
{
    float3 v = a - b;
    return dot(v, v);
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ float distance(const float3 &a, const float3 &b)
{
    return sqrtf(sqrDistance(a, b));
}
//--------------------------------------------------------------------------------------------------------------------//
inline __host__ __device__ float fract(float x)
{
    return x - floorf(x);
}
//--------------------------------------------------------------------------------------------------------------------//
