#pragma once

#include "defines.h"

class Context
{
private:
    Context();

public:
    ~Context() = default;
    static Context &getInstance();
    static cl::Device &D() { return getInstance().device; }
    static cl::Context &get() { return getInstance().context; }
    static cl::CommandQueue &Q() { return getInstance().queue; }
    static void PrintDevices();
    static void PrintDevice(const cl::Device &device);

private:
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    std::vector<cl::Platform> m_platforms;
    std::map<int, std::vector<cl::Device>> m_devices;
};
