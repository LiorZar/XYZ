#include "Program.h"

NAMESPACE_BEGIN(gl);

GLProgram::GLProgram(const std::string &vertexShaderSource, const std::string &fragmentShaderSource)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    const char *vertexShaderSourceCStr = vertexShaderSource.c_str();
    const char *fragmentShaderSourceCStr = fragmentShaderSource.c_str();

    glShaderSource(vertexShader, 1, &vertexShaderSourceCStr, NULL);
    glShaderSource(fragmentShader, 1, &fragmentShaderSourceCStr, NULL);

    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    GLint success;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (success == GL_FALSE)
    {
        GLint logLength;
        glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &logLength);
        std::vector<GLchar> errorLog(logLength);
        glGetShaderInfoLog(vertexShader, logLength, nullptr, errorLog.data());
        throw std::runtime_error("Vertex shader compilation failed: " + std::string(errorLog.data()));
    }

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (success == GL_FALSE)
    {
        GLint logLength;
        glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &logLength);
        std::vector<GLchar> errorLog(logLength);
        glGetShaderInfoLog(fragmentShader, logLength, nullptr, errorLog.data());
        throw std::runtime_error("Fragment shader compilation failed: " + std::string(errorLog.data()));
    }

    programID = glCreateProgram();
    glAttachShader(programID, vertexShader);
    glAttachShader(programID, fragmentShader);
    glLinkProgram(programID);

    glGetProgramiv(programID, GL_LINK_STATUS, &success);
    if (success == GL_FALSE)
    {
        GLint logLength;
        glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &logLength);
        std::vector<GLchar> errorLog(logLength);
        glGetProgramInfoLog(programID, logLength, nullptr, errorLog.data());
        throw std::runtime_error("Program linking failed: " + std::string(errorLog.data()));
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

GLProgram::~GLProgram()
{
    glDeleteProgram(programID);
}

void GLProgram::use() const
{
    glUseProgram(programID);
}

GLuint GLProgram::get() const
{
    return programID;
}
NAMESPACE_END(gl);