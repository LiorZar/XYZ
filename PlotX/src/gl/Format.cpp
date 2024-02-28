#include "Format.h"

NAMESPACE_BEGIN(gl);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
Attribute::Attribute(const std::string &_name, const GLuint _size, const GLenum _type)
    : name(_name), size(_size), type(_type), location(-1), defaultLocation(-1), pointer(nullptr)
{
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Attribute::Enable(const GLsizei _stride) const
{
    if (location < 0)
        return;

    glEnableVertexAttribArray(location);
    if (GL_FLOAT == type)
        glVertexAttribPointer(location, size, type, GL_FALSE, _stride, pointer);
    else
        glVertexAttribIPointer(location, size, type, _stride, pointer);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Attribute::Disable() const
{
    if (location < 0)
        return;

    glDisableVertexAttribArray(location);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
Format::Format(const std::string &_format, const GLsizei _stride)
    : format(_format), stride(_stride)
{
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
GLsizei Format::SizeOf(const std::string &_format)
{
    Format f(_format, 0);
    f.Create();
    return f.stride;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Format::Create()
{
    Clear();

    GLchar *pointer = 0;
    GLsizei strideSize = 0;
    GLuint attributeSize = 0;

    Attribute attr;
    char attFormat[32];
    std::istringstream ss(format);

    while (ss.getline(attFormat, 32, ','))
    {
        if (false == CreateAttribute(attFormat, attr, attributeSize))
            return false;

        attr.pointer = pointer;
        pointer += attributeSize;
        strideSize += attr.size;

        attributes.push_back(attr);
    }

    if (stride <= 0)
        stride = strideSize;

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Format::Clear()
{
    attributes.clear();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Format::EnableAttributes() const
{
    for (const auto &attribute : attributes)
        attribute.Enable(stride);
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Format::DisableAttributes() const
{
    for (const auto &attribute : attributes)
        attribute.Disable();
    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Format::ZeroAttributes()
{
    for (auto &attribute : attributes)
        attribute.location = -1;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Format::SetFormat(const std::string &_format)
{
    format = _format;
    Clear();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Format::FixByCurrentProgram(GLint programID)
{
    bool oneAttributeCreated = false;
    for (auto &attribute : attributes)
    {
        if (attribute.location < 0)
            attribute.location = programID ? glGetAttribLocation(programID, attribute.name.c_str()) : -1;

        if (attribute.location < 0)
        {
            if (attribute.defaultLocation < 0)
                continue;

            attribute.location = attribute.defaultLocation;
        }
        oneAttributeCreated = true;
    }
    return oneAttributeCreated = true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Format::CreateAttribute(const std::string &_format, Attribute &_attribute, GLuint &_sizeInBytes) const
{
    _sizeInBytes = 0;
    std::string nameStr, typeStr, idxStr;

    auto i0 = _format.find_first_of(':');
    if (std::string::npos == i0)
        return false;

    auto i1 = _format.find_first_of(':', i0 + 1);
    if (std::string::npos == i1)
        i1 = _format.length();

    nameStr = _format.substr(0, i0);
    typeStr = _format.substr(i0 + 1, i1 - (i0 + 1));
    if (i1 < _format.length())
        idxStr = _format.substr(i1 + 1);

    if (2 != typeStr.length())
        return false;

    GLenum type = 0;
    GLint length = typeStr[0] - '0';

    switch (typeStr[1])
    {
    case 'b':
        type = GL_BYTE;
        _sizeInBytes = sizeof(char);
        break;
    case 'B':
        type = GL_UNSIGNED_BYTE;
        _sizeInBytes = sizeof(unsigned char);
        break;
    case 's':
        type = GL_SHORT;
        _sizeInBytes = sizeof(short);
        break;
    case 'S':
        type = GL_UNSIGNED_SHORT;
        _sizeInBytes = sizeof(unsigned short);
        break;
    case 'i':
        type = GL_INT;
        _sizeInBytes = sizeof(int);
        break;
    case 'I':
        type = GL_UNSIGNED_INT;
        _sizeInBytes = sizeof(unsigned int);
        break;
    case 'f':
        type = GL_FLOAT;
        _sizeInBytes = sizeof(float);
        break;
    default:
        type = 0;
        break;
    }

    if (0 == type || length <= 0 || length > 4)
        return false;

    _sizeInBytes *= length;

    _attribute = Attribute(nameStr, length, type);
    _attribute.defaultLocation = _attribute.location = idxStr.length() > 0 ? atoi(idxStr.c_str()) : -1;

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(gl);