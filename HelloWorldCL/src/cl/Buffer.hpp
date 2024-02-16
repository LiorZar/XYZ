#pragma once

#include "Context.h"

template <typename T>
class Buffer
{
public:
    Buffer() = default;

    explicit Buffer(size_t size) : _data(size), dirty(true) {}
    void ClearDirty()
    {
        dirty = false;
    }
    void SetDirty()
    {
        dirty = true;
    }
    void Resize(size_t size)
    {
        _data.resize(size);
        dirty = true;
    }
    void Refresh() const
    {
        Upload();
    }
    void Upload() const
    {
        if (dirty)
        {
            if (_data.size() > gpuSize)
            {
                buffer = cl::Buffer(Context::get(), CL_MEM_READ_WRITE, sizeof(T) * _data.size());
                gpuSize = _data.size();
            }
            Context::Q().enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(T) * _data.size(), _data.data());
            dirty = false;
        }
    }
    void UploadSub(size_t offset, size_t size)
    {
        if (dirty)
        {
            Context::Q().enqueueWriteBuffer(buffer, CL_TRUE, sizeof(T) * offset, sizeof(T) * size, _data.data() + offset);
        }
    }
    void UploadAt(size_t offset)
    {
        if (dirty)
        {
            Context::Q().enqueueWriteBuffer(buffer, CL_TRUE, sizeof(T) * offset, sizeof(T) * _data.size(), _data.data() + offset);
        }
    }
    void DownloadSub(size_t offset, size_t size)
    {
        Context::Q().enqueueReadBuffer(buffer, CL_TRUE, sizeof(T) * offset, sizeof(T) * size, _data.data() + offset);
    }
    void DownloadAt(size_t offset)
    {
        Context::Q().enqueueReadBuffer(buffer, CL_TRUE, sizeof(T) * offset, sizeof(T) * _data.size(), _data.data() + offset);
    }

    void Download()
    {
        Context::Q().enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(T) * _data.size(), _data.data());
    }
    size_t size() const
    {
        return _data.size();
    }
    T &operator[](size_t index)
    {
        dirty = true;
        return _data[index];
    }
    const T &operator[](size_t index) const
    {
        return _data[index];
    }
    const cl::Buffer &d() const { return device(); }
    const cl::Buffer &device() const
    {
        Refresh();
        return buffer;
    }
    const std::vector<T> &host() const { return _data; }

    T *data()
    {
        dirty = true;
        return _data.data();
    }
    const T *data() const { return _data.data(); }

private:
    std::vector<T> _data;

    mutable bool dirty;
    mutable size_t gpuSize = 0;
    mutable cl::Buffer buffer;
};
