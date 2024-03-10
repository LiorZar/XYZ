#ifndef __FILE_H__
#define __FILE_H__

#include "defines.h"

class File
{
public:
    File(const std::string &path);

    bool Load(const std::string &path);

private:
    static bool file2String(const std::string &path, std::string &sourceCode);
    static bool file2String(const std::string &dir, std::string &file, std::string &sourceCode);
    static bool processIncludes(const std::string &dir, std::string &sourceCode, std::map<int, std::string> &lineMap);

public:
    std::string filepath;
    std::string sourceCode;
    std::map<int, std::string> fileByLine;

private:
    static std::set<std::string> s_dirs;
};

#endif // __FILE_H__