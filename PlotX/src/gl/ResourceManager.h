#ifndef __RESOURCE_MANAGER_H__
#define __RESOURCE_MANAGER_H__

#include "Program.h"

NAMESPACE_BEGIN(gl);

class ResourceManeger
{
public:
    ResourceManeger();
    virtual ~ResourceManeger();

public:
    bool Init(const std::string &dir);

protected:
    bool LoadShaders(const std::string &dir);

protected:
    std::string m_workDir;
    std::map<std::string, ProgramPtr> m_programs;
};

NAMESPACE_END(gl);

#endif // __RESOURCE_MANAGER_H__