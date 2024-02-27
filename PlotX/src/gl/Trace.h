#ifndef __TRACE_H__
#define __TRACE_H__

#include "defines.h"
#define PrintGLError gl::Trace::CheckGLError(__FILE__, __LINE__)
#define PrintLog(...) gl::Trace::Log(__FILE__, __LINE__, __VA_ARGS__)
#define PrintError(...) gl::Trace::Error(__FILE__, __LINE__, __VA_ARGS__)
#define PrintWarning(...) gl::Trace::Warning(__FILE__, __LINE__, __VA_ARGS__)

NAMESPACE_BEGIN(gl);

class Trace
{
public:
    static bool CheckGLError(const char *_file, const int _line);

public:
    static void Log(const char *_file, const int _line, const char *format, ...);
    static void Error(const char *_file, const int _line, const char *format, ...);
    static void Warning(const char *_file, const int _line, const char *format, ...);
};

NAMESPACE_END(gl);

#endif // __TRACE_H__