#pragma once

#include "defines.h"
NAMESPACE_BEGIN(utils);

void Trim(std::string &str);
int LineCount(const std::string &str, int offset = 0, int end = -1);
int Split(const std::string &str, std::vector<std::string> &vec, char delim = ' ');
int SplitTrim(const std::string &str, std::vector<std::string> &vec, char delim = ' ');

void ToCSV(const std::string &filename, const float *data, int size, int cols);

template <typename T>
void FromFile(const std::string &filename, std::vector<T> &data)
{
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open())
    {
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        data.resize(size / sizeof(T));
        file.read((char *)data.data(), size);
        file.close();
    }
}
NAMESPACE_END(utils);