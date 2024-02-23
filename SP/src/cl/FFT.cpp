#include "FFT.h"

//-------------------------------------------------------------------------------------------------------------------------------------------------//
cl::Buffer FFT::null;
//-------------------------------------------------------------------------------------------------------------------------------------------------//
FFT::Plan::Plan(size_t size, clfftLayout layout, clfftResultLocation _placeness, clfftPrecision _precision)
    : precision(_precision), placeness(_placeness)
{
    dims = CLFFT_1D;
    sizes[0] = size;
    iLayout = oLayout = layout;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
FFT::Plan::Plan(size_t sizex, size_t sizey, clfftLayout layout, clfftResultLocation _placeness, clfftPrecision _precision)
    : precision(_precision), placeness(_placeness)
{
    dims = CLFFT_2D;
    sizes[0] = sizex;
    sizes[1] = sizey;
    iLayout = oLayout = layout;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
FFT::Plan::Plan(size_t sizex, size_t sizey, size_t sizez, clfftLayout layout, clfftResultLocation _placeness, clfftPrecision _precision)
    : precision(_precision), placeness(_placeness)
{
    dims = CLFFT_3D;
    sizes[0] = sizex;
    sizes[1] = sizey;
    sizes[2] = sizez;
    iLayout = oLayout = layout;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
FFT::Plan::~Plan()
{
    if (handle != 0)
        clfftDestroyPlan(&handle);
    handle = 0;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
#define CHECK(cmd)              \
    err = cmd;                  \
    if(CLFFT_SUCCESS != err){   \
        std::cerr << #cmd << err << std::endl; \
    return false;               \
    }
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool FFT::Plan::Init()
{
    if (handle != 0)
        return true;
    clfftStatus err;
    CHECK(clfftCreateDefaultPlan(&handle, Context::get()(), dims, sizes));
    CHECK(clfftSetPlanBatchSize(handle, batchSize));
    CHECK(clfftSetPlanPrecision(handle, precision));
    CHECK(clfftSetResultLocation(handle, placeness));
    CHECK(clfftSetLayout(handle, iLayout, oLayout));
    if(dist > 0){
        CHECK(clfftSetPlanDistance(handle, dist, dist));
    }
    CHECK(clfftBakePlan(handle, 1, &Context::Q()(), nullptr, nullptr));

    size_t tmpBufferSize;
    clfftGetTmpBufSize(handle, &tmpBufferSize);
    if(tmpBufferSize > 0)
        tmpBuffer = cl::Buffer(Context::get(), CL_MEM_READ_WRITE, tmpBufferSize);

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
std::string FFT::Plan::Key() const
{
    static auto fn = [](clfftLayout layout)
    {
        switch (layout)
        {
        case CLFFT_COMPLEX_INTERLEAVED:
            return 'i';
        case CLFFT_COMPLEX_PLANAR:
            return 'p';
        case CLFFT_REAL:
            return 'r';
        }
        return ' '; // error
    };
    std::stringstream ss;
    ss << dims << "D_[";
    switch (dims)
    {
    case CLFFT_1D:
        ss << sizes[0] << "]";
        break;
    case CLFFT_2D:
        ss << sizes[0] << "x" << sizes[1] << "]";
        break;
    case CLFFT_3D:
        ss << sizes[0] << "x" << sizes[1] << "x" << sizes[2] << "]";
        break;
    }
    ss
        << (CLFFT_SINGLE == precision ? "f" : "d")
        << (CLFFT_INPLACE == placeness ? "i" : "o")
        << "_" << fn(iLayout) << fn(oLayout)
        << "_b" << batchSize << "_d" << dist;

    return ss.str();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
FFT::FFT()
{
    clfftSetupData fftSetup;
    auto err0 = clfftInitSetupData(&fftSetup);
    auto err1 = clfftSetup(&fftSetup);

    std::cout << "clfftInitSetupData: " << err0 << std::endl;
    std::cout << "clfftSetup: " << err1 << std::endl;

    PlanPtr plan;
//    plan = std::make_shared<Plan>(NextPow235(20499), CLFFT_COMPLEX_INTERLEAVED, CLFFT_OUTOFPLACE);
//    plan->Init();
//    plans[plan->Key()] = plan;

//    plan = std::make_shared<Plan>(NextPow235(20499), CLFFT_COMPLEX_INTERLEAVED, CLFFT_INPLACE);
//    plan->Init();
//    plans[plan->Key()] = plan;

//    plan = std::make_shared<Plan>(NextPow235(20999), CLFFT_COMPLEX_INTERLEAVED, CLFFT_INPLACE);
//    plan->Init();
//    plans[plan->Key()] = plan;

    plan = std::make_shared<Plan>(NextPow2(20999), CLFFT_COMPLEX_INTERLEAVED, CLFFT_INPLACE);
    plan->Init();
    plans[plan->Key()] = plan;

    plan = std::make_shared<Plan>(20, CLFFT_COMPLEX_INTERLEAVED, CLFFT_INPLACE);
    plan->batchSize = 20000;
    plan->dist = 0;
    plan->Init();
    plans[plan->Key()] = plan;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
FFT::~FFT()
{
    plans.clear();
    clfftTeardown();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
FFT &FFT::getInstance()
{
    static FFT instance;
    return instance;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
size_t FFT::NextPow2(size_t n)
{
    size_t p = 1;
    while (p < n)
        p <<= 1;
    return p;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
size_t FFT::NextPow235(size_t n)
{
    std::vector<unsigned long> numbers = {1};
    size_t i = 0, j = 0, k = 0; // Indices for 2, 3, and 5 multiples

    while (numbers.back() < n)
    {
        size_t next2 = numbers[i] * 2;
        size_t next3 = numbers[j] * 3;
        size_t next5 = numbers[k] * 5;

        size_t nextNumber = std::min<size_t>({next2, next3, next5});
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
int FFT::Dispatch(bool fwd, cl::Buffer &inputBuffer, cl::Buffer &outputBuffer, size_t size, size_t split, size_t dist)
{
    Plan plan(size, CLFFT_COMPLEX_INTERLEAVED, outputBuffer() != null() ? CLFFT_OUTOFPLACE : CLFFT_INPLACE);
    if(split > 1)
    {
        plan.batchSize = split;
        plan.sizes[0] /= split;
        if(dist > 0)
            plan.dist = dist;
    }
    return getInstance().dispatch(plan, fwd, inputBuffer, outputBuffer);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
int FFT::dispatch(const Plan &_plan, bool fwd, cl::Buffer &inputBuffer, cl::Buffer &outputBuffer)
{
    try
    {
        auto &plan = plans[_plan.Key()];
        if (plan == nullptr)
        {
            plan = std::make_shared<Plan>(_plan);
            if (false == plan->Init())
                throw std::runtime_error("Failed to initialize FFT plan");
        }
        cl::Event event;
        auto &queue = Context::Q();

        auto status = clfftEnqueueTransform(plan->handle, fwd ? CLFFT_FORWARD : CLFFT_BACKWARD, 1, &queue(), 0, NULL, &event(), &inputBuffer(), &outputBuffer(), plan->tmpBuffer());

#ifdef TIMING
        // queue.finish();
        event.wait();
        auto e = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        auto s = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        return int((e - s) / 1000);
#endif
    }
    catch (const std::exception &e)
    {
        std::cerr << "****************" << e.what() << '\n';
    }

    return 0;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
