#pragma once

const int WORK = 256;

#ifndef _USE_NV_CUDA // 1 = RUNTIME_COMPILATION
#define KERNEL(x) std::string(#x)
#define LaunchKernel(kernel, ...) Dispatch(KERNEL(kernel), __VA_ARGS__, CudaProgram::END_ARG)
#else
#define KERNEL(x) x
#define COMPILE_NV_CU
#define LaunchKernel(kernel, ...) DispatchNV(KERNEL(kernel), __VA_ARGS__, CudaProgram::END_ARG)
#endif


