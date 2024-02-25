#pragma once

#include "GPU.h"

NAMESPACE_BEGIN(cu);

class FFT
{
private:
    struct Plan
    {
        Plan(const Plan &) = default;
        Plan(int size, cufftType type = CUFFT_C2C);
        Plan(int sizex, int sizey, cufftType type = CUFFT_C2C);
        Plan(int sizex, int sizey, int sizez, cufftType type = CUFFT_C2C);
        ~Plan();

        bool Init();
        std::string Key() const;
        cufftHandle &operator()() { return handle; }

        cufftHandle handle = 0;
        cufftType type = CUFFT_C2C;
        int dims = 1;
        int sizes[3] = {0, 0, 0};
        int batchSize = 1;
        int dist = 0;
    };
    using PlanPtr = std::shared_ptr<Plan>;
    std::map<std::string, PlanPtr> plans;

private:
    FFT();
    ~FFT();

public:
    static FFT &getInstance();
    static int NextPow2(int n);
    static int NextPow235(int n);

public:
    template <typename T, typename G>
    static int Dispatch(bool fwd, T *inputBuffer, G *outputBuffer, int size, int split, int dist = 0)
    {
        Plan plan(size);
        if (sizeof(T) != sizeof(G))
        {
            if (sizeof(T) < sizeof(G))
                plan.type = CUFFT_C2R;
            else
                plan.type = CUFFT_R2C;
        }
        if (split > 1)
        {
            plan.batchSize = split;
            plan.sizes[0] /= split;
            if (dist > 0)
                plan.dist = dist;
        }
        return getInstance().dispatch(plan, fwd, inputBuffer, outputBuffer);
    }
    template <typename T>
    static int Dispatch(bool fwd, gbuffer<T> &inputBuffer, int split)
    {
        return Dispatch(fwd, *inputBuffer, *inputBuffer, (int)inputBuffer.size(), split);
    }
    template <typename T, typename G>
    static int Dispatch(bool fwd, gbuffer<T> &inputBuffer, gbuffer<G> &outputBuffer, int split)
    {
        return Dispatch(fwd, *inputBuffer, *outputBuffer, (int)inputBuffer.size(), split);
    }
    template <typename T>
    static int Dispatch(bool fwd, gbuffer<T> &inputBuffer, int offset, int size)
    {
        T *p = inputBuffer.p(offset);
        return Dispatch(fwd, p, p, size, 1);
    }
    template <typename T, typename G>
    static int Dispatch(bool fwd, gbuffer<T> &inputBuffer, gbuffer<G> &outputBuffer, int offset, int size)
    {
        return Dispatch(fwd, inputBuffer.p(offset), outputBuffer.p(offset), size, 1);
    }

private:
    int dispatch(const Plan &_plan, float *inputBuffer, float2 *outputBuffer);
    int dispatch(const Plan &_plan, float2 *inputBuffer, float *outputBuffer);
    int dispatch(const Plan &_plan, bool fwd, float2 *inputBuffer, float2 *outputBuffer);
};

NAMESPACE_END(cu);