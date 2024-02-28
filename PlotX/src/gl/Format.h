#ifndef __FORMAT_H__
#define __FORMAT_H__

#include "defines.h"

NAMESPACE_BEGIN(gl);

struct Attribute
{
    Attribute(const std::string &name = "", GLuint size = 0, GLenum type = 0);

    void Enable(GLsizei stride) const;
    void Disable() const;

    std::string name;
    GLuint size;
    GLenum type;
    GLint location;
    GLint defaultLocation;
    const GLvoid *pointer;
};

class Format
{
public:
    Format(const std::string &format, GLsizei stride);
    static GLsizei SizeOf(const std::string &format);

public:
    bool Create();
    void Clear();
    bool EnableAttributes() const;
    bool DisableAttributes() const;
    void ZeroAttributes();
    bool FixByCurrentProgram(GLint programID);

public:
    const std::string &GetFormat() const { return format; }
    void SetFormat(const std::string &format);

private:
    bool CreateAttribute(const std::string &format, Attribute &attribute, GLuint &sizeInBytes) const;

private:
    GLsizei stride;
    std::string format;
    std::vector<Attribute> attributes;
};

NAMESPACE_END(gl);

#endif // __FORMAT_H__