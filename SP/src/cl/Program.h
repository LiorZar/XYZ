#pragma once

#include "Context.h"

class Program
{
public:
    Program(const std::string &sourceFilePath);
    bool Load(const std::string &sourceFilePath);

public:
    template <typename... Args>
    int Dispatch1D(const std::string &kernel, size_t grid, size_t block, Args &&...args)
    {
        return Dispatch(kernel, cl::NDRange(grid), cl::NDRange(block), std::forward<Args>(args)...);
    }
    template <typename... Args>
    int Dispatch2D(const std::string &kernel, const cl_uint2 &grid, const cl_uint2 &block, Args &&...args)
    {
        return Dispatch(kernel, cl::NDRange(grid.x, grid.y), cl::NDRange(block.x, block.y), std::forward<Args>(args)...);
    }
    template <typename... Args>
    int Dispatch3D(const std::string &kernel, const cl_uint3 &grid, const cl_uint3 &block, Args &&...args)
    {
        return Dispatch(kernel, cl::NDRange(grid.x, grid.y, grid.z), cl::NDRange(block.x, block.y, block.z), std::forward<Args>(args)...);
    }

private:
    template <typename... Args>
    int Dispatch(const std::string &name, const cl::NDRange &global, const cl::NDRange &local, Args &&...args)
    {
        cl::Kernel &kernel = getKernel(name);
        if (false == setKernelArguments(kernel, std::forward<Args>(args)...))
            return -1;
        return Dispatch(kernel, global, local);
    }
    int Dispatch(cl::Kernel &kernel, const cl::NDRange &_global, const cl::NDRange &local);

    template <typename... Args>
    bool setKernelArguments(cl::Kernel &kernel, Args &&...args)
    {
        if (!kernel())
            return false;

        cl_uint index = 0;
        (kernel.setArg(index++, std::forward<Args>(args)), ...);

        return true;
    }
    cl::Kernel &getKernel(const std::string &kernelName);
    cl::NDRange adjustGlobalAndLocalSize(const cl::NDRange &global, const cl::NDRange &local) const;

private:
    cl::Program m_program;
    std::map<std::string, cl::Kernel> m_kernelMap;

private:
    static cl::Kernel null;
};
