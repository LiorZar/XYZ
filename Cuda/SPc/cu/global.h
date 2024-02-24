#pragma once

#include "defines.h"

NAMESPACE_BEGIN(cu);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
cudaError sync();
void checkError(cudaError _err);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
template <typename T>
void Free(T *&device, bool host = false)
{
    if (false == host)
        checkError(cudaFree(device));
    else
        checkError(cudaFreeHost(device));
    device = nullptr;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
template <typename T>
void Alloc(T *&device, size_t numElements = 1, bool host = false)
{
    if (false == host)
        checkError(cudaMalloc(&device, sizeof(T) * numElements));
    else
        cudaHostAlloc(&device, sizeof(T) * numElements, cudaHostAllocMapped);
}
template <typename T>
T *Pointer(T *device)
{
    T *p = nullptr;
    checkError(cudaHostGetDevicePointer(&p, device, 0));
    return p;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
template <typename T>
void Upload(const T &host, T &device) { checkError(cudaMemcpyToSymbol(device, &host, sizeof(T))); }
template <typename T>
void Upload(const T &host, T *device) { checkError(cudaMemcpy(device, &host, sizeof(T), cudaMemcpyHostToDevice)); }
template <typename T>
void Upload(const T *host, T *device, size_t length) { checkError(cudaMemcpy(device, host, length * sizeof(T), cudaMemcpyHostToDevice)); }
template <typename T>
void UploadAt(const T *host, T *device, size_t offset) { checkError(cudaMemcpy(device + offset, host, sizeof(T), cudaMemcpyHostToDevice)); }
//-------------------------------------------------------------------------------------------------------------------------------------------------//
template <typename T>
void Download(const T &device, T &host) { checkError(cudaMemcpyFromSymbol(&host, device, sizeof(T))); }
template <typename T>
void Download(const T *device, T &host) { checkError(cudaMemcpy(&host, device, sizeof(T), cudaMemcpyDeviceToHost)); }
template <typename T>
void Download(const T *device, T *host, size_t length, size_t offset) { checkError(cudaMemcpy(host, device + offset, length * sizeof(T), cudaMemcpyDeviceToHost)); }
template <typename T>
void DownloadAt(const T *device, T *host, size_t offset) { checkError(cudaMemcpy(host, device + offset, sizeof(T), cudaMemcpyDeviceToHost)); }
//-------------------------------------------------------------------------------------------------------------------------------------------------//
template <typename T>
void CopyAt(T *dst, const T *src, size_t i, size_t j) { checkError(cudaMemcpy(dst + i, src + j, sizeof(T), cudaMemcpyDeviceToDevice)); }
//-------------------------------------------------------------------------------------------------------------------------------------------------//
template <typename T>
class gbuffer
{
public:
    bool m_host;
    size_t m_size;
    T *data = nullptr;
    T *hata = nullptr;
    mutable std::vector<T> temp;

public:
    gbuffer(bool _host = false, size_t _size = 0) : m_host(_host), m_size(_size)
    {
        resize(_size);
    }
    ~gbuffer()
    {
        clear();
    }
    void clear()
    {
        if (data)
        {
            cu::Free(data, m_host);
            m_size = 0;
            data = nullptr;
            hata = nullptr;
        }
    }
    T *p() { return data; }
    T *h() { return hata; }
    void swap(gbuffer<T>& buff)
    {
		std::swap(buff.m_host, m_host);
        std::swap(buff.m_size, m_size);
        std::swap(buff.data, data);
        std::swap(buff.hata, hata);
    }
    std::vector<T> &cpu() { return temp; }
    size_t size() const { return m_size; }
    size_t length() const { return m_size; }
    void resize(size_t _size)
    {
        clear();
        if (_size > 0)
        {
            cu::Alloc(data, _size, m_host);
            if (m_host)
                hata = cu::Pointer(data);
            m_size = _size;
        }
    }
    void reallocate(size_t _size)
    {
        if (_size <= m_size)
            return;
        resize(_size);
    }
    void Upload(const T *_data, size_t _count)
    {
        reallocate(_count);
        cu::Upload(_data, data, _count);
    }
    bool Download(std::vector<T> &_data, size_t _count, size_t _offset = 0) const
    {
        if (_offset + _count > m_size)
            return false;
        _data.resize(_count);
        if (false == m_host)
            cu::Download(data, _data.data(), _count, _offset);
        else
        {
            cu::sync();
            std::copy(hata + _offset, hata + _offset + _count, _data.data());
        }

        return true;
    }
    bool Download(T *_data, size_t _count, size_t _offset = 0) const
    {
        if (_offset + _count > m_size)
            return false;

        if (false == m_host)
            cu::Download(data, _data, _count);
        else
        {
            cu::sync();
            std::copy(hata, hata + _count, _data);
        }

        return true;
    }
    void UploadAt(size_t index, const T &v)
    {
        cu::UploadAt(&v, data, index);
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
    void RefreshUp(size_t count = size_t(-1))
    {
        if (count > temp.size())
            count = temp.size();
        if (count > m_size)
            reallocate(count);

        Upload(temp.data(), count);
    }
    void RefreshDown(size_t count = size_t(-1)) const
    {
        if (count > m_size)
            count = m_size;

        Download(temp, count);
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
            std::sort(temp.begin(), temp.begin() + count, pred);
            RefreshUp(count);
        }
    }
    T &operator[](int i)
    {
        if (m_host)
            return hata[i];
        return temp[i];
    }
    const T &operator[](int i) const
    {
        if (m_host)
            return hata[i];
        return temp[i];
    }

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