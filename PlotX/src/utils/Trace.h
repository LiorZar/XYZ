#ifndef __TRACE_H__
#define __TRACE_H__

#include "defines.h"
#define PrintGLError Trace::CheckGLError(__FILE__, __LINE__)
#define PrintLog(...) Trace::Log(__FILE__, __LINE__, __VA_ARGS__)
#define PrintError(...) Trace::Error(__FILE__, __LINE__, __VA_ARGS__)
#define PrintWarning(...) Trace::Warning(__FILE__, __LINE__, __VA_ARGS__)

class Trace
{
public:
    static bool CheckGLError(const char *_file, const int _line);

public:
    static void Log(const char *_file, const int _line, const char *format, ...);
    static void Error(const char *_file, const int _line, const char *format, ...);
    static void Warning(const char *_file, const int _line, const char *format, ...);
    static void Lines(const char *sourceCode);
};

#endif // __TRACE_H__