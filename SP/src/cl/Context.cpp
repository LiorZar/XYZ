#include "Context.h"
#include <iomanip>
//-------------------------------------------------------------------------------------------------------------------------------------------------//
Context::Context()
{
    cl::Device fastestDevice;
    cl_ulong fastestClockFrequency = 0;
    cl::Platform::get(&m_platforms);
    for (auto i = 0U; i < m_platforms.size(); ++i)
    {
        m_platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &m_devices[i]);
        for (const auto &device : m_devices[i])
        {
            cl_ulong clockFrequency;
            device.getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &clockFrequency);

            if (clockFrequency > fastestClockFrequency)
            {
                fastestClockFrequency = clockFrequency;
                fastestDevice = device;
            }
        }
    }

    PrintDevice(fastestDevice);
    device = fastestDevice;

    context = cl::Context(fastestDevice);
    cl_queue_properties properties[] = {CL_QUEUE_PROFILING_ENABLE, 0};
    queue = cl::CommandQueue(context, fastestDevice, properties);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
Context &Context::getInstance()
{
    static Context instance;
    return instance;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Context::PrintDevices()
{
    auto &c = getInstance();
    for (auto i = 0U; i < c.m_platforms.size(); ++i)
    {
        const auto &platform = c.m_platforms[i];
        std::string platformName;
        platform.getInfo(CL_PLATFORM_NAME, &platformName);
        std::cout << "Platform: " << platformName << std::endl;

        for (const auto &device : c.m_devices[i])
            PrintDevice(device);
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Context::PrintDevice(const cl::Device &device)
{
    std::vector<std::pair<std::string, cl_device_info>> stringAttributes = {
        {"Device Name", CL_DEVICE_NAME},
        {"Device Vendor", CL_DEVICE_VENDOR},
        {"Device Version", CL_DEVICE_VERSION}};

    std::vector<std::pair<std::string, cl_device_info>> integerAttributes = {
        {"Device Max Work Group Size", CL_DEVICE_MAX_WORK_GROUP_SIZE},
        {"Device Local Memory Size", CL_DEVICE_LOCAL_MEM_SIZE},
        {"Device Global Memory Size", CL_DEVICE_GLOBAL_MEM_SIZE},
        {"Device Compute Units", CL_DEVICE_MAX_COMPUTE_UNITS},
        {"Device Clock Frequency", CL_DEVICE_MAX_CLOCK_FREQUENCY}};

    for (const auto &attribute : stringAttributes)
    {
        std::string attributeValue;
        device.getInfo(attribute.second, &attributeValue);
        std::cout << attribute.first << ": " << attributeValue << std::endl;
    }

    for (const auto &attribute : integerAttributes)
    {
        cl_ulong attributeValue;
        device.getInfo(attribute.second, &attributeValue);

        if (attributeValue > 1024 * 1024 * 1024)
        {
            double valueInGB = static_cast<double>(attributeValue) / (1024 * 1024 * 1024);
            std::cout << attribute.first << ": " << std::fixed << std::setprecision(2) << valueInGB << "G" << std::endl;
        }
        else if (attributeValue > 1024 * 1024)
        {
            double valueInMB = static_cast<double>(attributeValue) / (1024 * 1024);
            std::cout << attribute.first << ": " << std::fixed << std::setprecision(2) << valueInMB << "M" << std::endl;
        }
        else if (attributeValue > 1024)
        {
            double valueInKB = static_cast<double>(attributeValue) / 1024;
            std::cout << attribute.first << ": " << std::fixed << std::setprecision(2) << valueInKB << "K" << std::endl;
        }
        else
        {
            std::cout << attribute.first << ": " << attributeValue << std::endl;
        }
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
