#pragma once

#include "defines.h"
NAMESPACE_BEGIN(cu);

class Utils
{
public:
    static void Trim(std::string &str);
    static int Split(const std::string &str, std::vector<std::string> &vec, char delim = ' ');
    static int SplitTrim(const std::string &str, std::vector<std::string> &vec, char delim = ' ');

    static void ToCSV(const std::string &filename, const float *data, int size, int cols);

    template <typename T>
    static bool FromFile(const std::string &filename, std::vector<T> &data)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
            return false;

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        data.resize(size / sizeof(T));
        file.read((char *)data.data(), size);
        file.close();

        return true;
    }
    template <typename T>
    static bool ToFile(const std::string &filename, const std::vector<T> &data)
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
            return false;

        file.write((const char *)data.data(), data.size() * sizeof(T));
        file.close();

        return true;
    }
};

NAMESPACE_END(cu);