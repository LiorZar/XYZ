#include "Program.h"
#include "Trace.h"

NAMESPACE_BEGIN(gl);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
Program::Program()
{
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
Program::~Program()
{
    Destroy();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Program::Create(const std::vector<std::string> &files)
{
    std::vector<eShaderType> types;
    for (auto &file : files)
    {
        if (file.find(".vert") != std::string::npos)
            types.push_back(eShaderType::VERTEX);
        else if (file.find(".frag") != std::string::npos)
            types.push_back(eShaderType::FRAGMENT);
        else if (file.find(".geom") != std::string::npos)
            types.push_back(eShaderType::GEOM);
        else if (file.find(".comp") != std::string::npos)
            types.push_back(eShaderType::COMPUTE);
    }
    return create(files, types);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Program::Create(const std::string &filename, const std::vector<eShaderType> &types)
{
    std::vector<std::string> files;
    for (auto &type : types)
    {
        std::string file = filename;
        switch (type)
        {
        case eShaderType::VERTEX:
            file += ".vert";
            break;
        case eShaderType::FRAGMENT:
            file += ".frag";
            break;
        case eShaderType::GEOM:
            file += ".geom";
            break;
        case eShaderType::COMPUTE:
            file += ".comp";
            break;
        }
        files.push_back(file);
    }
    return create(files, types);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Program::Rebuild()
{
    if (programID)
        Destroy();
    return create(m_paths, m_types);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Program::Destroy()
{
    if (programID)
    {
        for (auto &shader : m_shaders)
            glDeleteShader(shader);
        m_shaders.clear();
        glDeleteProgram(programID);
        programID = 0;
    }
    PrintGLError;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Program::use() const
{
    glUseProgram(programID);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Program::SetSubroutine(eShaderType type, const std::string &name)
{
    auto itSub = m_subroutines.find(type);
    if (itSub == m_subroutines.end())
        return false;

    auto it2 = itSub->second.find(name);
    if (it2 == itSub->second.end())
        return false;

    auto itBlock = m_subroutinesIndexBlock.find(type);
    if (itBlock == m_subroutinesIndexBlock.end())
        return false;

    GLuint index = it2->second.first;
    GLint location = it2->second.second;
    itBlock->second[location] = index;

    glUniformSubroutinesuiv((GLuint)type, itBlock->second.size(), itBlock->second.data());

    m_activeSubroutine[type] = name;

    return PrintGLError;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Program::create(const std::vector<std::string> &files, const std::vector<eShaderType> &types)
{
    Destroy();

    m_paths = files;
    m_types = types;
    m_files.clear();

    programID = glCreateProgram();
    if (programID == 0)
    {
        PrintError("Failed to create program");
        return false;
    }

    int compiled;
    for (auto i = 0U; i < files.size(); ++i)
    {
        File file(files[i]);
        if (file.sourceCode.empty())
        {
            PrintError("Failed to load file: %s", files[i].c_str());
            Destroy();
            return false;
        }

        GLuint shader = glCreateShader((GLuint)types[i]);
        if (shader == 0)
        {
            PrintError("Failed to create shader: %s", files[i].c_str());
            Destroy();
            return false;
        }
        m_shaders.push_back(shader);

        const char *src = file.sourceCode.c_str();
        glShaderSource(shader, 1, &src, NULL);
        glCompileShader(shader);
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if (GL_FALSE == compiled)
        {
            int length = 0;
            int retLength = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
            if (length > 1)
            {
                GLchar *buffer = new GLchar[length + 1];
                glGetShaderInfoLog(shader, length, &retLength, buffer);
                PrintError("Failed to compile shader: %s", buffer);
                delete[] buffer;
                Trace::Lines(src);
                PrintGLError;
            }
            Destroy();
            return false;
        }
    }
    for (auto &shader : m_shaders)
        glAttachShader(programID, shader);

    PrintGLError;
    glLinkProgram(programID);
    glGetProgramiv(programID, GL_LINK_STATUS, &compiled);
    if (GL_TRUE != compiled)
    {
        PrintError("Failed to link program");
        Destroy();
        return false;
    }
    PrintGLError;

    extractUniforms();
    extractSubroutines();

    return PrintGLError;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Program::extractUniforms()
{
    m_uniforms.clear();

    GLint count;
    glGetProgramiv(programID, GL_ACTIVE_UNIFORMS, &count);
    for (auto i = 0; i < count; ++i)
    {
        char name[256];
        GLsizei length;
        GLint size;
        GLenum type;
        glGetActiveUniform(programID, i, 256, &length, &size, &type, name);
        m_uniforms[name] = glGetUniformLocation(programID, name);
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Program::extractSubroutines()
{
    m_subroutines.clear();
    m_subroutinesIndexBlock.clear();
    m_activeSubroutine.clear();

    for (const auto &type : m_types)
    {
        GLint count;
        glGetProgramStageiv(programID, (GLenum)type, GL_ACTIVE_SUBROUTINE_UNIFORMS, &count);
        if (count > 0)
        {
            IndexBlock indexBlock(count);
            // glGetIntegerv(GL_ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS, indexBlock.data());
            m_subroutinesIndexBlock[type] = indexBlock;

            SubroutineMap subroutineMap;
            for (auto i = 0; i < count; ++i)
            {
                char name[256];
                GLsizei length;
                GLint subCount, I;
                GLenum type;
                std::vector<GLint> subroutineID;
                glGetActiveSubroutineUniformName(programID, (GLenum)type, i, 256, &length, name);
                I = glGetSubroutineUniformLocation(programID, type, name);
                if (I < 0 || I >= count)
                    continue;
                glGetActiveSubroutineUniformiv(programID, (GLenum)type, i, GL_NUM_COMPATIBLE_SUBROUTINES, &subCount);
                if (subCount <= 0)
                    continue;
                subroutineID.resize(subCount);
                glGetActiveSubroutineUniformiv(programID, (GLenum)type, i, GL_COMPATIBLE_SUBROUTINES, subroutineID.data());
                for (auto &id : subroutineID)
                {
                    glGetActiveSubroutineName(programID, (GLenum)type, id, 256, &length, name);
                    if (length == 0)
                        continue;
                    SubroutineIdIndex subroutineIdIndex = {id, I};
                    subroutineIdIndex.first = id;
                    subroutineIdIndex.second = subCount;
                    subroutineMap[name] = subroutineIdIndex;
                }
                indexBlock[I] = subroutineID[0];
            }
            m_subroutines[type] = subroutineMap;
        }
        PrintGLError;
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(gl);