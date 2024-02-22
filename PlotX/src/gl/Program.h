#ifndef GL_PROGRAM_WRAPPER_H
#define GL_PROGRAM_WRAPPER_H

#include "defines.h"

class GLProgram
{
private:
    GLuint programID;

public:
    GLProgram(const std::string &vertexShaderSource, const std::string &fragmentShaderSource);
    ~GLProgram();

    void use() const;
    GLuint get() const;
};

#endif // GL_PROGRAM_WRAPPER_H