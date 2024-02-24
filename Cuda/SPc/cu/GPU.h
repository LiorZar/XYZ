#pragma once

#include "defines.h"
#include "global.h"
#include "buffer.hpp"

NAMESPACE_BEGIN(cu);

class GPU
{
private:
    GPU();
    GPU(const GPU &);

public:
    static GPU *Get();
    ~GPU();

public:
    typedef std::function<void(const char *_str)> TraceFn;
    static TraceFn sTraceFn;

public:
    static bool Set();
    static int Device();
    static __int64 CurrTime_us();
    static std::string GetCurrentDirectory() { return Get()->m_currDir; }
    static std::string GetWorkingDirectory() { return Get()->m_workDir; }

public:
    static void _cdecl trace(const char *lpszFormat, ...);
    static void _cdecl traceLines(const char *_lines);

private:
    bool init = false;
    int gpu = 0;
    std::string m_workDir;
    std::string m_currDir;
    std::string m_projDir;

private:
    static __int64 timeFreq;
};

NAMESPACE_END(cu);