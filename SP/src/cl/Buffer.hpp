#pragma once

#include "Context.h"

template <typename T>
std::vector<T> ReadFile(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << path << std::endl;
        return {};
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<T> data(size / sizeof(T));
    file.read(reinterpret_cast<char *>(data.data()), size);
    return data;
}

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
    bool ReadFromFile(const std::string &path, bool upload = true)
    {
        _data = ReadFile<T>(path);
        dirty = true;
        if (upload)
            Upload();
        return !_data.empty();
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
    cl::Buffer &operator*() { return device(); }
    const cl::Buffer &operator*() const { return device(); }
    const cl::Buffer &d() const { return device(); }
    cl::Buffer &device()
    {
        Refresh();
        return buffer;
    }
    const cl::Buffer &device() const
    {
        Refresh();
        return buffer;
    }
    std::vector<T> &operator&() { return host(); }
    const std::vector<T> &operator&() const { return host(); }

    std::vector<T> &host()
    {
        dirty = true;
        return _data;
    }
    const std::vector<T> &host() const { return _data; }

    T *data()
    {
        dirty = true;
        return _data.data();
    }
    const T *data() const { return _data.data(); }

    cl::Buffer &GetSubBuffer(size_t offset, size_t size) const
    {
        auto it = subBuffers.find({offset, size});
        if (it != subBuffers.end())
            return it->second;
        cl_buffer_region region;
        region.origin = sizeof(T) * offset;
        region.size = sizeof(T) * size;
        auto sub = buffer.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region);
        subBuffers[{offset, size}] = sub;
        return sub;
    }

private:
    std::vector<T> _data;

    mutable bool dirty;
    mutable size_t gpuSize = 0;
    mutable cl::Buffer buffer;
    mutable std::map<std::pair<size_t, size_t>, cl::Buffer> subBuffers;
};
