#ifndef __FILE_H__
#define __FILE_H__

#include "defines.h"

class File
{
private:
    using fpath = std::filesystem::path;

public:
    File(const std::string &path);
    bool Load(const std::string &path);

public:
    using FilesFn = std::function<void(const std::string &name, const std::vector<std::string> &files)>;

    static void Travel(const std::string &dir, const std::string &name, const std::vector<std::string> &exts, FilesFn fn);
    static void Travel(const std::string &dir, const std::string &name, const std::vector<std::string> &exts, std::vector<std::string> &files);

private:
    static bool file2String(const std::string &path, std::string &sourceCode);
    static bool file2StringSearch(const std::string &dir, std::string &file, std::string &sourceCode);
    static bool processIncludes(const std::string &dir, std::string &sourceCode, std::map<int, std::string> &lineMap);

public:
    std::string filepath;
    std::string sourceCode;
    std::map<int, std::string> fileByLine;

public:
    static std::set<std::string> s_dirs;
};

#endif // __FILE_H__