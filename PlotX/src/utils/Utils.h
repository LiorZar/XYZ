#pragma once

#include "defines.h"
NAMESPACE_BEGIN(utils);

void Trim(std::string &str);
int LineCount(const std::string &str, int offset = 0, int end = -1);
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

template <typename T>
bool Convert(const std::string &att, T &val)
{
    std::stringstream ss(att);
    ss >> val;

    return true;
}

template <typename T>
void Split(const std::string &str, std::vector<T> &vals, const char delim = ',')
{
    std::string item;
    std::stringstream ss(str);

    T t;
    while (std::getline(ss, item, delim))
    {
        Convert(item, t);
        vals.push_back(t);
    }
}

template <typename T>
void SplitTrim(const std::string &str, std::vector<T> &vals, const char delim = ',')
{
    std::string item;
    std::stringstream ss(str);

    T t;
    while (std::getline(ss, item, delim))
    {
        Trim(item);
        Convert(item, t);
        vals.push_back(t);
    }
}
NAMESPACE_END(utils);