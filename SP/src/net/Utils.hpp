
#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <sstream>
#include <map>

using XParams = std::vector<std::string>;
using XNode = std::map<std::string, std::string>;

class Utils
{
public:
    static XParams Split(const std::string &input, char delimiter)
    {
        XParams result;
        std::string token;
        std::istringstream tokenStream(input);
        while (std::getline(tokenStream, token, delimiter))
            result.push_back(token);
        return result;
    }
    static XParams Split(const std::string &input, const std::string &delimiter)
    {
        XParams result;
        size_t start = 0;
        size_t end = input.find(delimiter);
        while (end != std::string::npos)
        {
            result.push_back(input.substr(start, end - start));
            start = end + delimiter.length();
            end = input.find(delimiter, start);
        }
        result.push_back(input.substr(start, end));
        return result;
    }

    static std::string Join(const XParams &strings, const std::string &separator)
    {
        std::string result;
        for (size_t i = 0; i < strings.size(); ++i)
        {
            result += strings[i];
            if (i != strings.size() - 1)
            {
                result += separator;
            }
        }
        return result;
    }
};

#endif // UTILS_H