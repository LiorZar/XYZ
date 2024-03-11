#include "Utils.h"

NAMESPACE_BEGIN(utils);
//--------------------------------------------------------------------------------------------------------------------//
void Trim(std::string &str)
{
    str.erase(0, str.find_first_not_of(" \t\r\n"));
    str.erase(str.find_last_not_of(" \t\r\n") + 1);
}
//--------------------------------------------------------------------------------------------------------------------//
int LineCount(const std::string &str, int offset, int end)
{
    std::string s = str.substr(offset, end - offset + 1);
    return std::count(s.begin(), s.end(), '\n');
}
//--------------------------------------------------------------------------------------------------------------------//
void ToCSV(const std::string &filename, const float *data, int size, int cols)
{
    std::ofstream file(filename);
    for (int i = 0; i < size; i++)
    {
        file << data[i];
        if (i % cols == cols - 1)
            file << std::endl;
        else
            file << ", ";
    }
    file << std::endl;
}
//--------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(utils);