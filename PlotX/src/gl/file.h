#ifndef __FILE_H__
#define __FILE_H__

#include "defines.h"

NAMESPACE_BEGIN(gl);

class File
{
public:
    File(const std::string &path);

    bool Load(const std::string &path);

private:
    void processIncludes(std::string &sourceCode);

private:
    std::string filePath;
    std::string sourceCode;
    std::unordered_map<std::string, int> lineCounts;
    std::unordered_map<std::string, std::string> includedFiles;
};
NAMESPACE_END(gl);

#endif // __FILE_H__