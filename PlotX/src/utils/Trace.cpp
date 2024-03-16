#include "Trace.h"
#include <stdarg.h>

#define RESET "\033[0m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"

//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Trace::CheckGLError(const char *_file, const int _line)
{
    if (nullptr == glfwGetCurrentContext())
    {
        Log(_file, _line, "No OpenGL context available\n");
        return false;
    }

    bool rv = true;
    GLenum err;
    while (GL_NO_ERROR != (err = glGetError()))
    {
        rv = false;
        switch (err)
        {
        case GL_INVALID_ENUM:
            Error(_file, _line, "GL_INVALID_ENUM\n");
            break;
        case GL_INVALID_VALUE:
            Error(_file, _line, "GL_INVALID_VALUE\n");
            break;
        case GL_INVALID_OPERATION:
            Error(_file, _line, "GL_INVALID_OPERATION\n");
            break;
        case GL_STACK_OVERFLOW:
            Error(_file, _line, "GL_STACK_OVERFLOW\n");
            break;
        case GL_STACK_UNDERFLOW:
            Error(_file, _line, "GL_STACK_UNDERFLOW\n");
            break;
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            Error(_file, _line, "GL_INVALID_FRAMEBUFFER_OPERATION\n");
            break;
        case GL_OUT_OF_MEMORY:
            Error(_file, _line, "GL_OUT_OF_MEMORY\n");
            break;
        default:
            Error(_file, _line, "Unknown error %d\n", err);
            break;
        }
    }

    return rv;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Trace::Log(const char *_file, const int _line, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    printf(GREEN); // Set output color to green
    printf("%s(%d): ", _file, _line);
    vprintf(format, args);
    printf("\n");
    printf(RESET); // Reset output color
    va_end(args);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Trace::Error(const char *_file, const int _line, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    printf(RED); // Set output color to red
    fprintf(stderr, "%s(%d): ", _file, _line);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    printf(RESET); // Reset output color
    va_end(args);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Trace::Warning(const char *_file, const int _line, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    printf(YELLOW); // Set output color to yellow
    fprintf(stderr, "%s(%d): ", _file, _line);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    printf(RESET); // Reset output color
    va_end(args);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Trace::Lines(const char *sourceCode)
{
    int line = 1;
    std::string item;
    std::stringstream ss(sourceCode);

    while (std::getline(ss, item, '\n'))
    {
        printf("%d: %s\n", line, item.c_str());
        line++;
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
