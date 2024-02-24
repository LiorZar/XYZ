#pragma once
#include "global.h"

NAMESPACE_BEGIN(cu);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
template <typename T>
std::vector<T> ReadFile(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "**************** Failed to open file: " << path << std::endl;
        return {};
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<T> data(size / sizeof(T));
    file.read(reinterpret_cast<char *>(data.data()), size);
    return data;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
template <typename T>
class gbuffer
{
public:
    bool m_host;
    size_t m_size, m_gpuSize = 0;
    T *data = nullptr;
    T *hata = nullptr;
    mutable std::vector<T> temp;

public:
    gbuffer(size_t _size = 0, bool _host = false) : m_host(_host), m_size(_size)
    {
        resize(_size);
    }
    ~gbuffer()
    {
        clear();
    }
    T *p() { return data; }
    const T *p() const { return data; }
    T *operator*() { return data; }
    const T *operator*() const { return data; }

    T *h() { return hata; }
    const T *h() const { return hata; }
    T *operator&() { return hata; }
    const T *operator&() const { return hata; }

    size_t size() const { return m_size; }
    size_t length() const { return m_size; }

    bool ReadFromFile(const std::string &path, bool upload = true)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "**************** Failed to open file: " << path << std::endl;
            return false;
        }
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        size_t count = size / sizeof(T);
        reallocate(count);

        file.read(reinterpret_cast<char *>(hata), size);
        if (upload)
            RefreshUp(count);
        return true;
    }
    void WriteToFile(const std::string &path)
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(hata), sizeof(T) * m_size);
    }

    void swap(gbuffer<T> &buff)
    {
        std::swap(buff.m_host, m_host);
        std::swap(buff.m_size, m_size);
        std::swap(buff.m_gpuSize, m_gpuSize);
        std::swap(buff.data, data);
        std::swap(buff.hata, hata);
        std::swap(buff.temp, temp);
    }

    void clear()
    {
        if (data)
        {
            cu::Free(data, m_host);
            m_size = 0;
            m_gpuSize = 0;
            data = nullptr;
            hata = nullptr;
            temp.clear();
        }
    }
    void resize(size_t _size)
    {
        clear();
        if (_size > 0)
        {
            cu::Alloc(data, _size, m_host);
            if (m_host)
                hata = cu::Pointer(data);
            else
            {
                temp.resize(_size);
                hata = temp.data();
            }
            m_size = _size;
            m_gpuSize = _size;
        }
    }
    void reallocate(size_t _size)
    {
        if (_size <= m_gpuSize)
        {
            m_size = _size;
            return;
        }
        resize(_size);
    }
    void RefreshUp(size_t count = size_t(-1))
    {
        if (m_host)
            return;

        if (count > temp.size())
            count = temp.size();
        if (count > m_size)
            reallocate(count);

        cu::Upload(hata, data, count);
    }
    void RefreshDown(size_t count = size_t(-1))
    {
        if (m_host)
            return;
        if (count > m_size)
            count = m_size;
        cu::Download(data, hata, count);
    }
    void Upload(const T *_data, size_t count)
    {
        reallocate(count);
        cu::Upload(_data, data, count);
    }
    void Upload(const std::vector<T> &_data)
    {
        Upload(_data.data(), _data.size());
    }
    void UploadAt(size_t index, const T &v)
    {
        cu::UploadAt(&v, data, index);
    }
    void Download(T *_data, size_t count) const
    {
        if (count > m_size)
            count = m_size;
        cu::Download(data, _data, count);
    }
    void Download(std::vector<T> &_data) const
    {
        if (_data.size() != m_size)
            _data.resize(m_size);
        cu::Download(data, _data.data(), m_size);
    }
    T DownloadAt(size_t index) const
    {
        T v;
        if (false == m_host)
            cu::DownloadAt(data, &v, index);
        else
        {
            cu::sync();
            std::copy(hata, hata + index, &v);
        }
        return v;
    }
    void Sort(size_t count, std::function<bool(const T &a, const T &b)> pred)
    {
        if (count > m_size)
            count = m_size;

        if (m_host)
            std::sort(hata, hata + count, pred);
        else
        {
            RefreshDown(count);
            std::sort(hata, hata + count, pred);
            RefreshUp(count);
        }
    }
    T &operator[](int i) { return hata[i]; }
    const T &operator[](int i) const { return hata[i]; }

    T *begin() { return hata; }
    const T *begin() const { return hata; }
    T *end(size_t _count)
    {
        if (_count > m_size)
            _count = m_size;
        return hata + _count;
    }
    const T *end(size_t _count) const
    {
        if (_count > m_size)
            _count = m_size;
        return hata + _count;
    }
};
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(cu);