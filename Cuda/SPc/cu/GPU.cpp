#include "GPU.h"
#include <cstring>
#include <stdarg.h>
NAMESPACE_BEGIN(cu);

GPU::TraceFn GPU::sTraceFn = nullptr;
//-------------------------------------------------------------------------------------------------------------------------------------------------//
GPU::GPU()
{
    int count = 0;
    cudaError err = cudaGetDeviceCount(&count);

    gpu = 0;
    // gpu = count - 1;

    //     auto res = cuInit(0);
    //     if (cudaSuccess == res)
    init = true;

#if _WIN32 || _WIN64
    char buff[MAX_PATH];
    ::GetCurrentDirectory(MAX_PATH, buff);
    m_currDir = buff;
    GetModuleFileName(nullptr, buff, MAX_PATH);
    m_workDir = buff;
    m_workDir = m_workDir.substr(0, m_workDir.find_last_of("\\/") + 1);

    m_projDir = m_currDir.substr(0, m_currDir.find_last_of("\\/") + 1);
#endif
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
GPU::~GPU()
{
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
GPU *GPU::Get()
{
    static GPU instance;
    return &instance;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool GPU::Set()
{
    auto gpu = Get()->gpu;
    cudaError err = cudaSetDevice(gpu);

    return (cudaSuccess == err);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
int GPU::Device()
{
    return Get()->gpu;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void __cdecl GPU::trace(const char *lpszFormat, ...)
{
    // Compile to no op when not needed
    // #if defined  _DEBUG || defined FORCE_DEBUG_OUTPUT
    {
        va_list args;
        va_start(args, lpszFormat);

        char szBuffer[1024] = {0};
        size_t formatLen = strlen(lpszFormat);
        size_t bufferSize = sizeof(szBuffer);
        bool legalSize = (formatLen < bufferSize);

        if (legalSize)
            sprintf(szBuffer, lpszFormat, args);

        szBuffer[1023] = 0;
        printf(szBuffer);
        if (sTraceFn)
            sTraceFn(szBuffer);
        va_end(args);

        if (false == legalSize)
        {
            printf(lpszFormat);
            if (sTraceFn)
                sTraceFn(lpszFormat);
        }
    }
    // #endif
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void __cdecl GPU::traceLines(const char *_lines)
{
    // Compile to no op when not needed
    // #if defined  _DEBUG || defined FORCE_DEBUG_OUTPUT
    {
        int lineNumber = 1, len;
        char szBuffer[1024] = {0};
        const char *next = _lines;
        for (const char *line = strchr(next, '\n'); line; line = strchr(next, '\n'))
        {
            len = int(line - next);
            sprintf(szBuffer, "%3d %.*s\n", lineNumber, len, next);
            printf(szBuffer);
            if (sTraceFn)
            {
                sTraceFn(szBuffer);
            }
            ++lineNumber;
            next = line + 1;
        }
        if (next)
        {
            printf(szBuffer, "%3d %s\n", lineNumber, next);
        }
    }
    // #endif
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(cu);