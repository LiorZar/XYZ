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
void Copy(T *dst, const T *src, size_t size) { checkError(cudaMemcpy(dst, src, sizeof(T)*size, cudaMemcpyDeviceToDevice)); }
template <typename T>
void CopyAt(T *dst, const T *src, size_t i, size_t j) { checkError(cudaMemcpy(dst + i, src + j, sizeof(T), cudaMemcpyDeviceToDevice)); }
//-------------------------------------------------------------------------------------------------------------------------------------------------//
template <typename T>
void Zero(T *dst, size_t size) { checkError(cudaMemset(dst, 0, sizeof(T)*size)); }
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(cu);