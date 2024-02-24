#pragma once

#include "defines.h"
NAMESPACE_BEGIN(cu);

class Utils
{
public:
    static void Trim(std::string &str);
    static int Split(const std::string &str, std::vector<std::string> &vec, char delim = ' ');
    static int SplitTrim(const std::string &str, std::vector<std::string> &vec, char delim = ' ');
};

NAMESPACE_END(cu);