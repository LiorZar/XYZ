#ifndef __BUFFER_H__
#define __BUFFER_H__

#include "Format.h"

NAMESPACE_BEGIN(gl);

class Buffer
{
public:
    Buffer(int _elmSize, GLenum _type, GLenum _usage, const std::string &_format = "", GLint _base = 0);
    Buffer(const Buffer &buffer) = delete;
    void operator=(const Buffer &buffer) = delete;
    virtual ~Buffer();

public:
    GLuint operator()() const { return m_id; }
    bool IsValid() const { return m_id > 0; }
    bool Create();
    bool CreateFix(GLsizei elementCount);
    bool CreateFixBytes(GLsizei sizeInBytes);
    bool Destory();
    bool CopyTo(Buffer &buffer) const;
    bool SetFormat(const std::string &_format);
    size_t GetSize() const { return m_gpuSizeInBytes; }
    size_t Length() const { return GetSize() / m_elmSize; }

public:
    bool Render(GLenum primitive, int startIndex, GLsizei count);
    bool Render(GLint programID, GLenum primitive, GLint startIndex, GLsizei count);

public:
    bool EnableAttributes();
    bool EnableAttributes(GLuint _progHandle);
    bool PureEnableAttributes() const { return m_format.EnableAttributes(); }
    bool DisableAttributes() const { return m_format.DisableAttributes(); }
    void ZeroAttributes() { return m_format.ZeroAttributes(); }

public:
    void SetType(GLenum _type) { m_type = _type; }
    void SetUsage(GLenum _usage) { m_usage = _usage; }
    void SetBase(GLint _base) { m_base = _base; }

public:
    bool AllocateBytes(GLsizei sizeInBytes, const void *data = nullptr);
    bool AllocateFixedBytes(GLsizei sizeInBytes, const void *data = nullptr);
    bool UploadBytes(GLsizei offset, GLsizei sizeInBytes, const void *data);
    bool DownloadBytes(GLsizei offset, GLsizei sizeInBytes, void *data) const;
    template <typename T>
    bool Allocate(GLsizei elementCount, const T *data = nullptr) { return Allocate(sizeof(T) * elementCount, data); }
    template <typename T>
    bool AllocateFixed(GLsizei elementCount, const T *data = nullptr) { return AllocateFixed(sizeof(T) * elementCount, data); }
    template <typename T>
    bool Allocate(const std::vector<T> &data) { return Allocate(data.size(), data.data()); }
    template <typename T>
    bool AllocateFixed(const std::vector<T> &data) { return AllocateFixed(data.size(), data.data()); }
    template <typename T>
    bool Upload(GLsizei offset, GLsizei elementCount, const T *data) { return UploadBytes(offset * sizeof(T), sizeof(T) * elementCount, data); }
    template <typename T>
    bool Upload(const std::vector<T> data) { return Upload(0, data.size(), data.data()); }
    template <typename T>
    bool Download(GLsizei offset, GLsizei elementCount, T *data) { return DownloadBytes(offset * sizeof(T), sizeof(T) * elementCount, data); }
    template <typename T>
    bool Download(std::vector<T> &data) { return Download(0, data.size(), data.data()); }
    template <typename T>
    bool UploadAt(int i, const T &val) { return Upload(i, 1, &val); }
    template <typename T>
    T DownloadAt(int i) const
    {
        T val;
        Download(i, 1, &val);
        return val;
    }

public:
    bool Bind(GLint base = -1) const { return BindType(0, base); }
    bool BindType(GLenum _type, GLint _base = -1) const;
    bool Unbind() const { return BindType(0); }
    bool UnBindType(GLenum _type) const;
    bool BindRange(GLintptr offset, GLsizeiptr size, GLint _base = -1) const { return BindRangeType(offset, size, 0, _base); }
    bool BindRangeType(GLintptr offset, GLsizeiptr size, GLenum _type, GLint _base = -1) const;

protected:
    GLuint m_id;
    GLenum m_type;
    GLenum m_usage;
    GLint m_base;

protected:
    GLsizei m_gpuSizeInBytes;

protected:
    const int m_elmSize;
    Format m_format;
};

NAMESPACE_END(gl);

#endif // __BUFFER_H__
