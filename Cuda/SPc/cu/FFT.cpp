#include "FFT.h"

NAMESPACE_BEGIN(cu);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
#define CHECK(cmd)                            \
    err = cmd;                                \
    if (CUFFT_SUCCESS != err)                 \
    {                                         \
        std::cerr << "err in " << #cmd        \
                  << " " << err << std::endl; \
        return false;                         \
    }
//-------------------------------------------------------------------------------------------------------------------------------------------------//
FFT::Plan::Plan(int size, cufftType type)
    : type(type),
      dims(1)
{
    sizes[0] = size;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
FFT::Plan::Plan(int sizex, int sizey, cufftType type)
    : type(type),
      dims(2)
{
    sizes[0] = sizex;
    sizes[1] = sizey;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
FFT::Plan::Plan(int sizex, int sizey, int sizez, cufftType type)
    : type(type),
      dims(3)
{
    sizes[0] = sizex;
    sizes[1] = sizey;
    sizes[2] = sizez;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
FFT::Plan::~Plan()
{
    if (handle != 0)
        cufftDestroy(handle);
    handle = 0;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool FFT::Plan::Init()
{
    if (handle != 0)
        return true;
    cufftResult err;

    if (batchSize > 1)
    {
        int dist = 0;
        switch (dims)
        {
        case 1:
            dist = sizes[0];
            CHECK(cufftPlanMany(&handle, 1, sizes, nullptr, 1, dist, nullptr, 1, dist, type, batchSize));
            break;
        case 2:
            dist = sizes[0] * sizes[1];
            CHECK(cufftPlanMany(&handle, 2, sizes, nullptr, 1, dist, nullptr, 1, dist, type, batchSize));
            break;
        case 3:
            dist = sizes[0] * sizes[1] * sizes[2];
            CHECK(cufftPlanMany(&handle, 3, sizes, nullptr, 1, dist, nullptr, 1, dist, type, batchSize));
            break;
        }
    }
    else
    {
        switch (dims)
        {
        case 1:
            CHECK(cufftPlan1d(&handle, sizes[0], type, batchSize));
            break;
        case 2:
            CHECK(cufftPlan2d(&handle, sizes[0], sizes[1], type));
            break;
        case 3:
            CHECK(cufftPlan3d(&handle, sizes[0], sizes[1], sizes[2], type));
            break;
        }
    }
    cufftGetSize(handle, &workSize);

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
std::string FFT::Plan::Key() const
{
    std::stringstream ss;
    ss << dims << "D_[";
    for (int i = 0; i < dims; ++i)
        ss << sizes[i] << "x";
    ss << "]_";
    switch (type)
    {
    case CUFFT_C2C:
        ss << "C2C";
        break;
    case CUFFT_R2C:
        ss << "R2C";
        break;
    case CUFFT_C2R:
        ss << "C2R";
        break;
    }
    ss << "_b" << batchSize << "_d" << dist;
    return ss.str();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
int FFT::NextPow2(int n)
{
    int p = 1;
    while (p < n)
        p <<= 1;
    return p;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
int FFT::NextPow235(int n)
{
    std::vector<int> numbers = {1};
    int i = 0, j = 0, k = 0; // Indices for 2, 3, and 5 multiples

    while (numbers.back() < n)
    {
        int next2 = numbers[i] * 2;
        int next3 = numbers[j] * 3;
        int next5 = numbers[k] * 5;

        int nextNumber = std::min<int>({next2, next3, next5});
        numbers.push_back(nextNumber);

        if (nextNumber == next2)
            ++i;
        if (nextNumber == next3)
            ++j;
        if (nextNumber == next5)
            ++k;
    }

    return numbers.back();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
FFT::FFT()
{
    PlanPtr plan;
    plan = std::make_shared<Plan>(NextPow2(20999), CUFFT_C2C);
    plan->Init();
    AddPlan(plan);

    plan = std::make_shared<Plan>(20, CUFFT_C2C);
    plan->batchSize = 20000;
    plan->dist = 0;
    plan->Init();
    AddPlan(plan);

	plan = std::make_shared<Plan>(1024, CUFFT_C2C);
	plan->batchSize = 38;
	plan->dist = 0;
	plan->Init();
	AddPlan(plan);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
FFT::~FFT()
{
    if(workArea)
        cudaFree(workArea);
    plans.clear();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
FFT &FFT::Get()
{
    static FFT instance;
    return instance;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
int FFT::dispatch(const Plan &_plan, float *inputBuffer, float2 *outputBuffer)
{
    cufftResult err;
    auto &plan = plans[_plan.Key()];
	if(nullptr == plan)
	{
        plan = std::make_shared<Plan>(_plan);
        if (false == plan->Init())
            return -1;
    }

    CHECK(cufftExecR2C(plan->handle, inputBuffer, outputBuffer));

    return 0;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
int FFT::dispatch(const Plan &_plan, float2 *inputBuffer, float *outputBuffer)
{
    cufftResult err;
    auto &plan = plans[_plan.Key()];
	if(nullptr == plan)
	{
        plan = std::make_shared<Plan>(_plan);
        if (false == plan->Init())
            return -1;
    }

    CHECK(cufftExecC2R(plan->handle, inputBuffer, outputBuffer));

    return 0;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
int FFT::dispatch(const Plan &_plan, bool fwd, float2 *inputBuffer, float2 *outputBuffer)
{
    cufftResult err;
    auto &plan = plans[_plan.Key()];
    if (nullptr == plan)
    {
        plan = std::make_shared<Plan>(_plan);
        if (false == plan->Init())
            return -1;
        AddPlan(plan);
    }

    if (fwd)
    {
        CHECK(cufftExecC2C(plan->handle, inputBuffer, outputBuffer, CUFFT_FORWARD));
    }
    else
    {
        CHECK(cufftExecC2C(plan->handle, inputBuffer, outputBuffer, CUFFT_INVERSE));
    }

    return 0;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void FFT::AddPlan(PlanPtr plan)
{
    bool didSizeChange = false;
    if(maxWorkSize < plan->workSize)
    {
        didSizeChange = true;
        maxWorkSize = plan->workSize;
        if(workArea)
            cudaFree(workArea);
        cudaMalloc(&workArea, maxWorkSize);
    }
    plans[plan->Key()] = plan;
    if(didSizeChange)
    {
        for(auto& [_,p]: plans)
            cufftSetWorkArea(p->handle, workArea);
    }
    else
        cufftSetWorkArea(plan->handle, workArea);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//

NAMESPACE_END(cu);