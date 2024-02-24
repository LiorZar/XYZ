#include "Utils.h"

NAMESPACE_BEGIN(cu);
//--------------------------------------------------------------------------------------------------------------------//
void Utils::Trim(std::string &str)
{
    str.erase(0, str.find_first_not_of(" \t\r\n"));
    str.erase(str.find_last_not_of(" \t\r\n") + 1);
}
//--------------------------------------------------------------------------------------------------------------------//
int Utils::Split(const std::string &str, std::vector<std::string> &vec, char delim)
{
    vec.clear();
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delim))
        vec.push_back(item);
    return (int)vec.size();
}
//--------------------------------------------------------------------------------------------------------------------//
int Utils::SplitTrim(const std::string &str, std::vector<std::string> &vec, char delim)
{
    vec.clear();
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        Trim(item);
        vec.push_back(item);
    }
    return (int)vec.size();
}
//--------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(cu);