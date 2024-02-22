#ifndef __PROGRAM_H__
#define __PROGRAM_H__

#include "defines.h"

NAMESPACE_BEGIN(gl);

class Program
{
private:
    GLuint programID;

public:
    Program(const std::string &vertexShaderSource, const std::string &fragmentShaderSource);
    ~Program();

    void use() const;
    GLuint get() const;
};
NAMESPACE_END(gl);

#endif // __PROGRAM_H__