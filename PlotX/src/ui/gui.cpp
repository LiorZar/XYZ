#include "gui.h"

NAMESPACE_BEGIN(ui);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
eHorzAlign ToHorzAlign(const std::string &str)
{
    static std::map<std::string, eHorzAlign> s_map =
        {
            {"left", eHorzAlign::Left},
            {"center", eHorzAlign::Center},
            {"right", eHorzAlign::Right},
        };
    auto it = s_map.find(str);
    if (s_map.end() != it)
        return it->second;

    return eHorzAlign::Left;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
eVertAlign ToVertAlign(const std::string &str)
{
    static std::map<std::string, eVertAlign> s_map =
        {
            {"top", eVertAlign::Top},
            {"middle", eVertAlign::Middle},
            {"bottom", eVertAlign::Bottom},
        };
    auto it = s_map.find(str);
    if (s_map.end() != it)
        return it->second;

    return eVertAlign::Top;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
eLayout ToLayout(const std::string &str)
{
    static std::map<std::string, eLayout> s_map =
        {
            {"vertical", eLayout::Vertical},
            {"horizontal", eLayout::Horizontal}};
    auto it = s_map.find(str);
    if (s_map.end() != it)
        return it->second;

    return eLayout::None;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
float CalcChildXPosition(const rectangle &_container, const rectangle &_child, eHorzAlign _align, const rectangle &_padding, float _offset)
{
    switch (_align)
    {
    case eHorzAlign::Left:
        return _child.x + _padding.x + _offset;
    case eHorzAlign::Center:
        return (_container.Width() - _child.Width()) * 0.5f + _child.x + _offset;
    case eHorzAlign::Right:
        return _container.Width() - _child.Width() - _child.x - _padding.z - _offset;
    default:
        return 0;
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
float CalcChildYPosition(const rectangle &_container, const rectangle &_child, eVertAlign _align, const rectangle &_padding, float _offset)
{
    switch (_align)
    {
    case eVertAlign::Top:
        return _child.y + _padding.y + _offset;
    case eVertAlign::Middle:
        return (_container.Height() - _child.Height()) * 0.5f + _child.y + _offset;
    case eVertAlign::Bottom:
        return _container.Height() - _child.Height() - _child.y - _padding.w - _offset;
    default:
        return 0;
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(ui);