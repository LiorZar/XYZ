#include "ResourceManager.h"

using namespace std;

NAMESPACE_BEGIN(gl);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
ResourceManeger::ResourceManeger()
{
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
ResourceManeger::~ResourceManeger()
{
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool ResourceManeger::Init(const string &dir)
{
    m_workDir = dir;
    File::s_dirs.insert(dir + "shaders/");
    bool rv = true;
    rv = LoadShaders(dir + "shaders") && rv;

    return rv;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool ResourceManeger::LoadShaders(const string &dir)
{
    bool rv = true;
    File::Travel(dir, "", {".vert", ".frag", ".geom"}, [&](const string &folderName, const vector<string> &files)
                 {
        bool root = dir.empty();
        string ext, path, name;
        map<string, vector<string>> extsMap;
        for (const auto &filename : files)
        {
            auto file = filesystem::path(filename);
            ext = file.extension().string();
            if (ext.empty())
                continue;
                
            path = file.replace_extension().string();
            extsMap[path].push_back(ext);
        }

        vector<eShaderType> types;
        for (const auto &[path, exts] : extsMap)
        {
            types.clear();
            for (const auto &ext : exts)
            {
                if (".vert" == ext)
                    types.push_back(eShaderType::VERTEX);
                else if (".frag" == ext)
                    types.push_back(eShaderType::FRAGMENT);
                else if (".geom" == ext)
                    types.push_back(eShaderType::GEOM);
            }
            if (types.empty())
                continue;

            ProgramPtr program = make_shared<Program>();
            if (program->Create(path, types))
            {
                name = root ? path : folderName + "." + path;
                m_programs[name] = program;
            }
        } });
    return false;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(gl);
