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
NAMESPACE_END(ui);