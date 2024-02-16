#define CL_TARGET_OPENCL_VERSION 210

#include <stdlib.h>
#include <CL/cl.h>
#include <iostream>
#include <vector>

int main()
{
    int x = sizeof(void *);
    std::cout << "Size of void*: " << x << std::endl;
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);

    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);

    for (const auto &platform : platforms)
    {
        char platformName[128];
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platformName, nullptr);
        std::cout << "Platform: " << platformName << std::endl;

        cl_uint deviceCount = 0;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);

        std::vector<cl_device_id> devices(deviceCount);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);

        for (const auto &device : devices)
        {
            char deviceName[128];
            clGetDeviceInfo(device, CL_DEVICE_NAME, 128, deviceName, nullptr);
            std::cout << " Device: " << deviceName << std::endl;
        }
    }

    return 0;
}
