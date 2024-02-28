#include "Buffer.h"
#include "Trace.h"

NAMESPACE_BEGIN(gl);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
Buffer::Buffer(int _elmSize, GLenum _type, GLenum _usage, const std::string &_format, GLint _base)
    : m_id(0), m_elmSize(_elmSize), m_type(_type), m_usage(_usage), m_base(_base), m_gpuSizeInBytes(0), m_format(_format, 0)
{
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
Buffer::~Buffer()
{
    Destory();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Buffer::Create()
{
    Destory();

    glGenBuffers(1, &m_id);
    return m_id > 0;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Buffer::CreateFix(GLsizei elementCount)
{
    return Create() && AllocateFixedBytes(m_elmSize * elementCount, nullptr);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Buffer::CreateFixBytes(GLsizei sizeInBytes)
{
    return Create() && AllocateFixedBytes(sizeInBytes, nullptr);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Buffer::Destory()
{
    if (m_id)
    {
        glDeleteBuffers(1, &m_id);
        m_id = 0;
    }
    return PrintGLError;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Buffer::CopyTo(Buffer &buffer) const
{
    if (m_id == 0 || buffer.m_id == 0)
        return false;

    if (buffer.m_gpuSizeInBytes < m_gpuSizeInBytes)
        buffer.AllocateBytes(m_gpuSizeInBytes);

    BindType(GL_COPY_READ_BUFFER);
    buffer.BindType(GL_COPY_WRITE_BUFFER);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, m_gpuSizeInBytes);
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    return PrintGLError;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Buffer::SetFormat(const std::string &_format)
{
    m_format.SetFormat(_format);
    if (IsValid())
        return Create();
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Buffer::Render(GLenum primitive, int startIndex, GLsizei count)
{
    return Render(-1, primitive, startIndex, count);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Buffer::Render(GLint programID, GLenum primitive, GLint startIndex, GLsizei count)
{
    if (false == Bind())
        return false;

    if (-1 == programID)
        glGetIntegerv(GL_CURRENT_PROGRAM, &programID);

    EnableAttributes(programID);
    glDrawArrays(primitive, startIndex, count);
    DisableAttributes();

    return Unbind();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Buffer::EnableAttributes()
{
    GLint programID = -1;
    glGetIntegerv(GL_CURRENT_PROGRAM, &programID);
    return EnableAttributes(programID);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Buffer::EnableAttributes(GLuint _progHandle)
{
    if (false == m_format.FixByCurrentProgram(_progHandle))
        return false;

    return m_format.EnableAttributes();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Buffer::AllocateBytes(GLsizei sizeInBytes, const void *data)
{
    if (false == Bind())
        return false;

    glBufferData(m_type, sizeInBytes, data, m_usage);
    m_gpuSizeInBytes = sizeInBytes;

    return PrintGLError;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Buffer::AllocateFixedBytes(GLsizei sizeInBytes, const void *data)
{
    if (false == Bind())
        return false;

    glBufferStorage(m_type, sizeInBytes, data, 0);
    m_gpuSizeInBytes = sizeInBytes;

    return PrintGLError;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Buffer::UploadBytes(GLsizei offset, GLsizei sizeInBytes, const void *data)
{
    if (false == Bind())
        return false;

    if (offset + sizeInBytes > m_gpuSizeInBytes)
        sizeInBytes = m_gpuSizeInBytes - offset;
    if (sizeInBytes <= 0)
        return true;

    glBufferSubData(m_type, offset, sizeInBytes, data);

    return PrintGLError;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Buffer::DownloadBytes(GLsizei offset, GLsizei sizeInBytes, void *data) const
{
    if (false == Bind())
        return false;

    if (offset + sizeInBytes > m_gpuSizeInBytes)
        sizeInBytes = m_gpuSizeInBytes - offset;
    if (sizeInBytes <= 0)
        return true;

    glGetBufferSubData(m_type, offset, sizeInBytes, data);

    return PrintGLError;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Buffer::BindType(GLenum type, GLint base) const
{
    PrintGLError;
    if (0 == type)
        type = m_type;
    if (-1 == base)
        base = m_base;

    switch (type)
    {
    case GL_UNIFORM_BUFFER:
    case GL_ATOMIC_COUNTER_BUFFER:
    case GL_SHADER_STORAGE_BUFFER:
        glBindBufferBase(type, base, m_id);
        break;

    default:
        glBindBuffer(type, m_id);
        break;
    }

    return PrintGLError;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Buffer::UnBindType(GLenum type) const
{
    if (0 == type)
        type = m_type;

    switch (type)
    {
    case GL_UNIFORM_BUFFER:
    case GL_ATOMIC_COUNTER_BUFFER:
    case GL_SHADER_STORAGE_BUFFER:
        glBindBufferBase(type, m_base, 0);
        break;

    default:
        glBindBuffer(type, 0);
        break;
    }

    return PrintGLError;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Buffer::BindRangeType(GLintptr offset, GLsizeiptr size, GLenum type, GLint base) const
{
    if (0 == type)
        type = m_type;
    if (-1 == base)
        base = m_base;

    glBindBufferRange(type, base, m_id, offset, size);

    return PrintGLError;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(gl);
