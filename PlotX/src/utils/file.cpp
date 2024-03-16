#include "file.h"

//-------------------------------------------------------------------------------------------------------------------------------------------------//
using namespace std;
//-------------------------------------------------------------------------------------------------------------------------------------------------//
set<string> File::s_dirs;
//-------------------------------------------------------------------------------------------------------------------------------------------------//
File::File(const string &path)
{
    Load(path);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool File::Load(const string &path)
{
    string srcCode;
    if (false == file2String(path, srcCode))
    {
        cerr << "Failed to open file: " << path << endl;
        return false;
    }
    string dir = path.substr(0, path.find_last_of("/\\") + 1);
    if (false == processIncludes(dir, srcCode, fileByLine))
    {
        cerr << "Failed to process includes: " << path << endl;
        return false;
    }
    filepath = path;
    sourceCode = srcCode;

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void File::Travel(const string &dir, const string &name, const vector<string> &exts, vector<string> &files)
{
    Travel(dir, name, exts, [&files](const string &name, const vector<string> &f)
           { files.insert(files.end(), f.begin(), f.end()); });
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void File::Travel(const string &dir, const string &name, const vector<string> &exts, FilesFn fn)
{
    const filesystem::path folder(dir);

    string filename, ext;
    vector<string> files;

    for (const auto &entry : filesystem::directory_iterator(folder))
    {
        filename = entry.path().filename().string();
        if ("." == filename || ".." == filename)
            continue;

        if (entry.is_directory())
        {
            Travel(dir + "/" + filename, (name.empty() ? "" : name + ".") + filename, exts, fn);
            continue;
        }
        if (false == exts.empty())
        {
            ext = entry.path().extension().string();
            if (exts.end() == find(exts.begin(), exts.end(), ext))
                continue;
        }
        files.push_back(dir + "/" + filename);
    }
    fn(name, files);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool File::file2String(const string &path, string &sourceCode)
{
    ifstream file(path);
    if (false == file.is_open())
        return false;

    stringstream buffer;
    buffer << file.rdbuf();
    sourceCode = buffer.str();

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool File::file2StringSearch(const string &dir, string &file, string &sourceCode)
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
bool File::processIncludes(const string &dir, string &sourceCode, map<int, string> &lineMap)
{
    size_t pos = sourceCode.find("#include");
    while (pos != string::npos)
    {
        if (pos > 0 && sourceCode[pos - 1] != '\n')
        {
            pos = sourceCode.find("#include", pos + 1);
            continue;
        }
        size_t start = sourceCode.find("\"", pos);
        size_t end = sourceCode.find("\"", start + 1);
        size_t newline = sourceCode.find("\n", start + 1);
        if (start != string::npos && end != string::npos && (newline == string::npos || newline > end))
        {
            int lineCount = count(sourceCode.begin(), sourceCode.begin() + start, '\n');
            string includeSourceCode;
            string includeFilePath = sourceCode.substr(start + 1, end - start - 1);
            if (false == file2StringSearch(dir, includeFilePath, includeSourceCode))
            {
                cerr << "Failed to open file: " << includeFilePath << endl;
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
