#include "Program.h"
#include <exception>

//-------------------------------------------------------------------------------------------------------------------------------------------------//
Program::Program(const std::string &sourceFilePath)
{
    Load(sourceFilePath);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Program::Load(const std::string &sourceFilePath)
{
    std::cout << "Current directory path: " << std::filesystem::current_path().string() << std::endl;

    // Read the source file
    std::ifstream sourceFile(sourceFilePath);
    if (!sourceFile.is_open())
    {
        std::cerr << "Failed to open source file: " << sourceFilePath << std::endl;
        return false;
    }

    static std::string s_pragmaStr = "#define __cl__\r\n";
    std::stringstream sourceStream;
    sourceStream << sourceFile.rdbuf();
    auto sourceStr = s_pragmaStr + sourceStream.str();

    // Create the OpenCL program
    cl::Program::Sources sources(1, std::make_pair(sourceFilePath.c_str(), sourceFilePath.length()));
    auto &context = Context::get();
    cl::Program program(context, sourceStr);

    // Build the program
    if (program.build() != CL_SUCCESS)
    {
        std::cerr << "Failed to build program" << std::endl;
        std::cerr << "Build log:" << std::endl;
        std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
        return false;
    }
    m_program = program;

    // Extract all kernels from the program
    std::vector<cl::Kernel> kernels;
    program.createKernels(&kernels);
    if (kernels.size() > 0)
        std::cout << "Kernels: " << kernels.size() << std::endl;
    for (const auto &k : kernels)
    {
        auto name = k.getInfo<CL_KERNEL_FUNCTION_NAME>();
        m_kernelMap[name] = k;
        std::cout << "\tKernel: " << name << std::endl;
    }

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
int Program::Dispatch(cl::Kernel &kernel, const cl::NDRange &_global, const cl::NDRange &local)
{
    try
    {
        auto global = adjustGlobalAndLocalSize(_global, local);
        cl::CommandQueue &queue = Context::Q();

        cl::Event event;
        auto res = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL, &event);
        if (res != CL_SUCCESS)
        {
            std::cerr << "Failed to enqueue kernel: " << res << std::endl;
            return -1;
        }
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
        std::cerr << "OpenCL error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
cl::Kernel Program::null;
//-------------------------------------------------------------------------------------------------------------------------------------------------//
cl::Kernel &Program::getKernel(const std::string &name)
{
    auto it = m_kernelMap.find(name);
    if (it == m_kernelMap.end())
    {
        std::cerr << "Kernel not found: " << name << std::endl;
        return null;
    }
    return it->second;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
cl::NDRange Program::adjustGlobalAndLocalSize(const cl::NDRange &global, const cl::NDRange &local) const
{
    size_t _global[3] = {global[0], global[1], global[2]};
    for (size_t i = 0; i < global.dimensions(); ++i)
    {
        size_t g = _global[i];
        size_t l = local[i];
        size_t r = g % l;
        if (r != 0)
            _global[i] += l - r;
    }
    switch (global.dimensions())
    {
    case 1:
        return cl::NDRange(_global[0]);
    case 2:
        return cl::NDRange(_global[0], _global[1]);
    }
    return cl::NDRange(_global[0], _global[1], _global[2]);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
