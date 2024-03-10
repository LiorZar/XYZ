#include "file.h"

//-------------------------------------------------------------------------------------------------------------------------------------------------//
std::set<std::string> File::s_dirs;
//-------------------------------------------------------------------------------------------------------------------------------------------------//
File::File(const std::string &path)
{
    Load(path);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool File::Load(const std::string &path)
{
    std::string srcCode;
    if (false == file2String(path, srcCode))
    {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }
    if (false == processIncludes(path, srcCode, fileByLine))
    {
        std::cerr << "Failed to process includes: " << path << std::endl;
        return false;
    }
    filepath = path;
    sourceCode = srcCode;
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool File::file2String(const std::string &path, std::string &sourceCode)
{
    std::ifstream file(path);
    if (false == file.is_open())
        return false;

    std::stringstream buffer;
    buffer << file.rdbuf();
    sourceCode = buffer.str();

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool File::file2String(const std::string &dir, std::string &file, std::string &sourceCode)
{
    if (file2String(dir + file, sourceCode))
    {
        file = dir + file;
        return true;
    }
    for (auto &dr : s_dirs)
    {
        if (file2String(dr + file, sourceCode))
        {
            file = dr + file;
            return true;
        }
    }
    return false;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool File::processIncludes(const std::string &dir, std::string &sourceCode, std::map<int, std::string> &lineMap)
{
    std::size_t pos = sourceCode.find("#include");
    while (pos != std::string::npos)
    {
        if (pos > 0 && sourceCode[pos - 1] != '\n')
        {
            pos = sourceCode.find("#include", pos + 1);
            continue;
        }
        std::size_t start = sourceCode.find("\"", pos);
        std::size_t end = sourceCode.find("\"", start + 1);
        std::size_t newline = sourceCode.find("\n", start + 1);
        if (start != std::string::npos && end != std::string::npos && (newline == std::string::npos || newline > end))
        {
            int lineCount = std::count(sourceCode.begin(), sourceCode.begin() + start, '\n');
            std::string includeSourceCode;
            std::string includeFilePath = sourceCode.substr(start + 1, end - start - 1);
            if (false == file2String(dir, includeFilePath, includeSourceCode))
            {
                std::cerr << "Failed to open file: " << includeFilePath << std::endl;
                return false;
            }
            lineMap[lineCount] = includeFilePath;
            sourceCode.replace(pos, end - pos + 1, includeSourceCode);
        }
        pos = sourceCode.find("#include", pos + 1);
    }
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
