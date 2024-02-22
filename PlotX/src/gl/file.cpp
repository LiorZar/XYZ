#include "file.h"

NAMESPACE_BEGIN(gl);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
File::File(const std::string &path) : filePath(filePath)
{
    Load(filePath);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void File::Load(const std::string &filePath)
{
    std::ifstream file(filePath);
    if (file.is_open())
    {
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string sourceCode = buffer.str();
        processIncludes(sourceCode);
        includedFiles[filePath] = sourceCode;
        lineCounts[filePath] = std::count(sourceCode.begin(), sourceCode.end(), '\n') + 1;
    }
    else
    {
        std::cerr << "Failed to open file: " << filePath << std::endl;
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//

void processIncludes(std::string &sourceCode)
{
    std::size_t pos = sourceCode.find("#include");
    while (pos != std::string::npos)
    {
        std::size_t start = sourceCode.find("\"", pos);
        std::size_t end = sourceCode.find("\"", start + 1);
        if (start != std::string::npos && end != std::string::npos)
        {
            std::string includeFilePath = sourceCode.substr(start + 1, end - start - 1);
            std::string includeSourceCode;
            loadFile(includeFilePath);
            auto it = includedFiles.find(includeFilePath);
            if (it != includedFiles.end())
            {
                includeSourceCode = it->second;
            }
            sourceCode.replace(pos, end - pos + 1, includeSourceCode);
        }
        pos = sourceCode.find("#include", pos + 1);
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(gl);
