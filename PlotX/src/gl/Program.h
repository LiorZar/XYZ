#ifndef __PROGRAM_H__
#define __PROGRAM_H__

#include "defines.h"
#include "file.h"

NAMESPACE_BEGIN(gl);

enum class eShaderType : GLuint
{
    VERTEX = GL_VERTEX_SHADER,
    FRAGMENT = GL_FRAGMENT_SHADER,
    GEOM = GL_GEOMETRY_SHADER,
    COMPUTE = GL_COMPUTE_SHADER
};

class Program
{
public:
    Program();
    virtual ~Program();

public:
    bool Create(const std::vector<std::string> &files);
    bool Create(const std::string &filename, const std::vector<eShaderType> &types);
    bool Rebuild();
    void Destroy();

public:
    void use() const;
    GLuint get() const;
    GLint operator[](const std::string &name) const { return m_uniforms.at(name); }
    bool SetSubroutine(eShaderType type, const std::string &name);

protected:
    bool create(const std::vector<std::string> &files, const std::vector<eShaderType> &types);

    void extractUniforms();
    void extractSubroutines();

private:
    GLuint programID = 0;
    std::vector<File> m_files;
    std::vector<std::string> m_paths;
    std::vector<eShaderType> m_types;
    std::vector<GLuint> m_shaders;
    std::map<std::string, GLuint> m_uniforms;

private:
    using IndexBlock = std::vector<GLuint>;
    using SubroutineIdIndex = std::pair<GLuint, GLint>;
    using SubroutineMap = std::map<std::string, SubroutineIdIndex>;

    std::map<eShaderType, IndexBlock> m_subroutinesIndexBlock;
    std::map<eShaderType, SubroutineMap> m_subroutines;
    std::map<eShaderType, std::string> m_activeSubroutine;
};
NAMESPACE_END(gl);

#endif // __PROGRAM_H__