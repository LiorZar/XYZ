#include "Xml.h"
#include "Utils.h"

NAMESPACE_BEGIN(Xml);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NodePtr Node::FromString(const std::string &str)
{
    NodePtr node = std::make_shared<Node>();
    if (0 == node->FromStr(str.c_str()))
        return nullptr;
    return node;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NodePtr Node::FromFile(const std::string &path)
{
    std::ifstream ifs(path);
    if (!ifs)
        return nullptr;

    std::stringstream oss;
    oss << ifs.rdbuf() << std::endl;
    ifs.close();

    return Node::FromString(oss.str());
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
Node::Node(const std::string &_type, NodePtr _parent, const std::string &_value) : m_type(_type),
                                                                                   m_value(_value),
                                                                                   m_parent(_parent)
{
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NodePtr Node::operator=(const Node &_node)
{
    NodePtr me = shared_from_this();
    if (&_node == this)
        return me;

    Clear();
    m_parent.reset();
    m_type = _node.m_type;
    m_value = _node.m_value;
    m_attributes = _node.m_attributes;
    for (const auto &child : _node.m_children)
    {
        NodePtr mychild = child->Clone();
        mychild->m_parent = me;
        m_children.push_back(mychild);
    }

    return me;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NodePtr Node::Clone() const
{
    auto result = std::make_shared<Node>();
    *result = *this;
    return result;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Node::Clear()
{
    m_attributes.clear();
    m_children.clear();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NodePtr Node::Merge(NodePtr _node)
{
    NodePtr me = shared_from_this();
    if (_node.get() == this)
        return me;

    m_type = _node->m_type;
    for (const auto [key, value] : _node->m_attributes)
        m_attributes.try_emplace(key, value);

    for (const auto child : _node->m_children)
    {
        const auto name = child->attr("name");
        auto mychild = ByName(name);
        if (mychild)
            mychild->Merge(child);
        else
            m_children.push_back(child->Clone());
    }

    return me;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
Node::~Node()
{
    Clear();
    m_parent.reset();
    m_value.clear();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Convert(const std::string &att, bul &val)
{
    val = ("1" == att || "true" == att);

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
template <int K, typename T>
bool Convert(const std::string &att, T *val)
{
    std::string item;
    std::stringstream ss(att);

    for (unsigned i = 0; i < K && std::getline(ss, item, ','); ++i)
        utils::Convert(item, val[i]);

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
template <int K, typename T>
bool getVec(const Node *node, const std::string &name, vec<K, T, defaultp> &val, const vec<K, T, defaultp> &def = vec<K, T, defaultp>())
{
    val = def;
    std::string res;
    if (!node->get(name, res))
        return false;
    vec4 v;
    Convert<K, T>(res, &val.x);

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
template <int K, typename T>
bool getVecX(const Node *node, const std::string &name, vec<K, T> &val)
{
    std::vector<T> data;
    node->getX(name, data);

    int S = std::min(K, (int)data.size());
    for (int i = 0; i < S; ++i)
        val[i] = data[i];

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
template <typename T>
bool getVector(const Node *node, const std::string &name, std::vector<T> &vals)
{
    std::string res;
    if (!node->get(name, res))
        return false;

    utils::Split(res, vals);

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::get(const std::string &name, bul &val, bul def) const
{
    std::string res;
    val = def;
    if (!get(name, res))
        return false;
    return utils::Convert(res, val);
}
bool Node::get(const std::string &name, s08 &val, s08 def) const
{
    std::string res;
    val = def;
    if (!get(name, res))
        return false;
    return utils::Convert(res, val);
}
bool Node::get(const std::string &name, u08 &val, u08 def) const
{
    std::string res;
    val = def;
    if (!get(name, res))
        return false;
    return utils::Convert(res, val);
}
bool Node::get(const std::string &name, s16 &val, s16 def) const
{
    std::string res;
    val = def;
    if (!get(name, res))
        return false;
    return utils::Convert(res, val);
}
bool Node::get(const std::string &name, u16 &val, u16 def) const
{
    std::string res;
    val = def;
    if (!get(name, res))
        return false;
    return utils::Convert(res, val);
}
bool Node::get(const std::string &name, s32 &val, s32 def) const
{
    std::string res;
    val = def;
    if (!get(name, res))
        return false;
    return utils::Convert(res, val);
}
bool Node::get(const std::string &name, u32 &val, u32 def) const
{
    std::string res;
    val = def;
    if (!get(name, res))
        return false;
    return utils::Convert(res, val);
}
bool Node::get(const std::string &name, s64 &val, s64 def) const
{
    std::string res;
    val = def;
    if (!get(name, res))
        return false;
    return utils::Convert(res, val);
}
bool Node::get(const std::string &name, u64 &val, u64 def) const
{
    std::string res;
    val = def;
    if (!get(name, res))
        return false;
    return utils::Convert(res, val);
}
bool Node::get(const std::string &name, f32 &val, f32 def) const
{
    std::string res;
    val = def;
    if (!get(name, res))
        return false;
    return utils::Convert(res, val);
}
bool Node::get(const std::string &name, f64 &val, f64 def) const
{
    std::string res;
    val = def;
    if (!get(name, res))
        return false;
    return utils::Convert(res, val);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::get(const std::string &name, std::string &val, const std::string &def) const
{
    val = def;
    auto it = m_attributes.find(name);
    if (m_attributes.end() == it)
        return false;

    val = it->second;

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
std::string Node::attr(const std::string &name, const std::string &def) const
{
    auto it = m_attributes.find(name);
    if (m_attributes.end() == it)
        return def;

    return it->second;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
vec4 Node::getColor(const std::string &name, const vec4 &def) const
{
    std::string val;
    if (!get(name, val) || val.empty() || val.length() < 7)
        return def;

    vec4 clr;
    if ('#' == val[0])
    {
        val = val.substr(1);
        clr.r = std::stoul(val.substr(0, 2), nullptr, 16) / 255.0f;
        clr.g = std::stoul(val.substr(2, 2), nullptr, 16) / 255.0f;
        clr.b = std::stoul(val.substr(4, 2), nullptr, 16) / 255.0f;
        clr.a = val.length() > 6 ? std::stoul(val.substr(6, 2), nullptr, 16) / 255.0f : 1.0f;
    }
    else
    {
        std::vector<f32> vals;
        utils::Split(val, vals);
        clr.r = vals.size() > 0 ? vals[0] : 0.0f;
        clr.g = vals.size() > 1 ? vals[1] : 0.0f;
        clr.b = vals.size() > 2 ? vals[2] : 0.0f;
        clr.a = vals.size() > 3 ? vals[3] : 1.0f;
    }
    return clr;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::get(const std::string &name, vec2 &val, const vec2 &def) const { return getVec(this, name, val, def); }
bool Node::get(const std::string &name, vec3 &val, const vec3 &def) const { return getVec(this, name, val, def); }
bool Node::get(const std::string &name, vec4 &val, const vec4 &def) const { return getVec(this, name, val, def); }
bool Node::get(const std::string &name, dvec2 &val, const dvec2 &def) const { return getVec(this, name, val, def); }
bool Node::get(const std::string &name, dvec3 &val, const dvec3 &def) const { return getVec(this, name, val, def); }
bool Node::get(const std::string &name, dvec4 &val, const dvec4 &def) const { return getVec(this, name, val, def); }
bool Node::get(const std::string &name, ivec2 &val, const ivec2 &def) const { return getVec(this, name, val, def); }
bool Node::get(const std::string &name, ivec3 &val, const ivec3 &def) const { return getVec(this, name, val, def); }
bool Node::get(const std::string &name, ivec4 &val, const ivec4 &def) const { return getVec(this, name, val, def); }
bool Node::get(const std::string &name, uvec2 &val, const uvec2 &def) const { return getVec(this, name, val, def); }
bool Node::get(const std::string &name, uvec3 &val, const uvec3 &def) const { return getVec(this, name, val, def); }
bool Node::get(const std::string &name, uvec4 &val, const uvec4 &def) const { return getVec(this, name, val, def); }
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::get(const std::string &name, std::vector<bul> &vals) const { return getVector(this, name, vals); }
bool Node::get(const std::string &name, std::vector<s08> &vals) const { return getVector(this, name, vals); }
bool Node::get(const std::string &name, std::vector<u08> &vals) const { return getVector(this, name, vals); }
bool Node::get(const std::string &name, std::vector<s16> &vals) const { return getVector(this, name, vals); }
bool Node::get(const std::string &name, std::vector<u16> &vals) const { return getVector(this, name, vals); }
bool Node::get(const std::string &name, std::vector<s32> &vals) const { return getVector(this, name, vals); }
bool Node::get(const std::string &name, std::vector<u32> &vals) const { return getVector(this, name, vals); }
bool Node::get(const std::string &name, std::vector<s64> &vals) const { return getVector(this, name, vals); }
bool Node::get(const std::string &name, std::vector<u64> &vals) const { return getVector(this, name, vals); }
bool Node::get(const std::string &name, std::vector<f32> &vals) const { return getVector(this, name, vals); }
bool Node::get(const std::string &name, std::vector<f64> &vals) const { return getVector(this, name, vals); }
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::get(const std::string &name, std::vector<std::string> &vals) const
{
    std::string res;
    if (!get(name, res))
        return false;

    std::string item;
    std::stringstream ss(res);

    while (std::getline(ss, item, ','))
        vals.push_back(item);

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::getX(const std::string &name, ivec2 &val) const { return getVecX(this, name, val); }
bool Node::getX(const std::string &name, ivec3 &val) const { return getVecX(this, name, val); }
bool Node::getX(const std::string &name, ivec4 &val) const { return getVecX(this, name, val); }
bool Node::getX(const std::string &name, uvec2 &val) const { return getVecX(this, name, val); }
bool Node::getX(const std::string &name, uvec3 &val) const { return getVecX(this, name, val); }
bool Node::getX(const std::string &name, uvec4 &val) const { return getVecX(this, name, val); }
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::getX(const std::string &name, std::vector<s32> &vals) const
{
    std::string res;
    if (!get(name, res))
        return false;

    std::string item;
    std::stringstream ss(res);

    while (std::getline(ss, item, ','))
        vals.push_back(std::stoul(item, nullptr, 16));

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::getX(const std::string &name, std::vector<u32> &vals) const
{
    std::string res;
    if (!get(name, res))
        return false;

    std::string item;
    std::stringstream ss(res);

    while (std::getline(ss, item, ','))
        vals.push_back(std::stoul(item, nullptr, 16));

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NodePtr Node::First(const std::string &type) const
{
    if (type.empty())
        return m_children.end() != m_children.begin() ? *m_children.begin() : nullptr;

    for (const auto &child : m_children)
    {
        if (type == child->Type())
            return child;
    }
    return nullptr;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NodePtr Node::Last(const std::string &type) const
{
    if (type.empty())
        return m_children.rend() != m_children.rbegin() ? *m_children.rbegin() : nullptr;

    for (auto it = m_children.rbegin(); m_children.rend() != it; it++)
    {
        if (type == (*it)->Type())
            return *it;
    }
    return nullptr;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NodePtr Node::Prev(const std::string &type) const
{
    const auto parent = m_parent.lock();
    if (nullptr == parent)
        return nullptr;

    const auto end = parent->m_children.rend();
    auto it = parent->m_children.rbegin();
    for (; end != it && this != it->get(); it++)
        ;
    if (end == it)
        return nullptr;

    it++;
    if (end == it)
        return nullptr;

    if (type.empty())
        return *it;

    for (; end != it; it++)
    {
        if (type == (*it)->Type())
            return *it;
    }

    return nullptr;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NodePtr Node::Next(const std::string &type) const
{
    const auto parent = m_parent.lock();
    if (nullptr == parent)
        return nullptr;

    const auto end = parent->m_children.end();
    auto it = parent->m_children.begin();
    for (; end != it && this != it->get(); it++)
        ;
    if (end == it)
        return nullptr;

    it++;
    if (end == it)
        return nullptr;

    if (type.empty())
        return *it;

    for (; end != it; it++)
    {
        if (type == (*it)->Type())
            return *it;
    }

    return nullptr;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NodePtr Node::ByAttr(const std::string &att, const std::string &value) const
{
    if (att.empty())
        return nullptr;

    for (const auto &child : m_children)
    {
        if (value == child->attr(att))
            return child;
    }
    return nullptr;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
std::vector<NodePtr> Node::Children(const std::string &type) const
{
    std::vector<NodePtr> childs;
    const bool noType = type.empty();

    for (const auto &child : m_children)
    {
        if (noType || type == child->Type())
            childs.push_back(child);
    }

    return childs;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::InsertEnd(NodePtr addThis)
{
    if (nullptr == addThis)
        return false;

    addThis->RemoveFromParent();
    addThis->m_parent = shared_from_this();
    m_children.push_back(addThis);

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::InsertBefore(NodePtr beforeThis, NodePtr addThis)
{
    if (nullptr == beforeThis || this != beforeThis->Parent().get() || nullptr == addThis)
        return false;

    addThis->RemoveFromParent();
    addThis->m_parent = shared_from_this();

    const auto end = m_children.end();
    const auto begin = m_children.begin();
    auto it = begin;
    for (; end != it && beforeThis != *it; it++)
        ;

    if (end == it || begin == it)
        m_children.push_front(addThis);
    else
    {
        it--;
        if (begin == it)
            m_children.push_front(addThis);
        else
            m_children.insert(it, addThis);
    }

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::InsertAfter(NodePtr afterThis, NodePtr addThis)
{
    if (nullptr == afterThis || this != afterThis->Parent().get() || nullptr == addThis)
        return false;

    addThis->RemoveFromParent();
    addThis->m_parent = shared_from_this();

    const auto end = m_children.end();
    auto it = m_children.begin();
    for (; end != it && afterThis != *it; it++)
        ;

    if (m_children.end() == it)
        m_children.push_back(addThis);
    else
        m_children.insert(it, addThis);

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::Replace(NodePtr replaceThis, NodePtr addThis)
{
    if (nullptr == replaceThis || this != replaceThis->Parent().get() || nullptr == addThis)
        return false;

    addThis->RemoveFromParent();
    addThis->m_parent = shared_from_this();

    const auto end = m_children.end();
    auto it = m_children.begin();
    for (; end != it && replaceThis != *it; it++)
        ;

    if (m_children.end() == it)
        m_children.push_back(addThis);
    else
    {
        m_children.insert(it, addThis);
        m_children.erase(it);
    }

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::Remove(NodePtr removeThis)
{
    if (nullptr == removeThis || this != removeThis->Parent().get())
        return false;

    const auto end = m_children.end();
    auto it = m_children.begin();
    for (; end != it && removeThis != *it; it++)
        ;
    if (end == it)
        return false;

    m_children.erase(it);
    removeThis->m_parent.reset();

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Node::RemoveFromParent()
{
    auto parent = Parent();
    if (nullptr == parent)
        return false;

    parent->Remove(shared_from_this());

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
inline static bool IsWhiteSpace(char c)
{
    return (iswspace(c) || c == '\n' || c == '\r');
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
inline static const char *SkipWhiteSpace(const char *p)
{
    if (!p || !*p)
        return 0;

    while (*p && IsWhiteSpace(*p) || *p == '\n' || *p == '\r')
        ++p;

    return p;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
inline static bool XStrEqual(const char *p, const char *tag)
{
    if (!p || !*p)
        return false;

    const char *q = p;

    while (*q && *tag && *q == *tag)
    {
        ++q;
        ++tag;
    }

    return (*tag == 0);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
inline static const char *ReadToEndTag(const char *p, const std::string &endTag)
{
    for (; p && *p; ++p)
    {
        if (0 == endTag.compare(0, endTag.size(), p, endTag.size()))
            break;
    }
    if (0 == p)
        return p;
    return p + endTag.length();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
#define ENTITY_COUNT 5
static const std::string entity[ENTITY_COUNT][2] =
    {
        {"&amp;", "&"},
        {"&lt;", "<"},
        {"&gt;", ">"},
        {"&quot;", "\""},
        {"&apos;", "\'"}};
//-------------------------------------------------------------------------------------------------------------------------------------------------//
inline static const char *ReadText(const char *p, std::string &text, char end)
{
    text = "";
    p = SkipWhiteSpace(p);
    for (; p && *p && *p != end; p++)
    {
        if ('&' != *p)
            text += *p;
        else
        {
            for (int i = 0; i < ENTITY_COUNT; ++i)
            {
                if (0 != strncmp(entity[i][0].c_str(), p, entity[i][0].length()))
                    continue;

                text += entity[i][1];
                p += (entity[i][0].length() - 1);
            }
        }
    }
    return p + 1;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
inline static const char *ReadNonXml(const char *p)
{
    if (!p || !*p)
        return 0;

    if (iswalpha(p[1]) || '_' == p[1])
        return p;

    static const std::string xmlHeader("<?");
    static const std::string xmlHeaderEnd(">");
    static const std::string commentHeader("<!--");
    static const std::string commentHeaderEnd("-->");
    static const std::string cdataHeader("<![CDATA[");
    static const std::string cdataHeaderEnd("]]>");
    static const std::string dtdHeader("<!");
    static const std::string dtdHeaderEnd(">");

    if (0 == xmlHeader.compare(0, xmlHeader.size(), p, xmlHeader.size()))
        return ReadToEndTag(p, xmlHeaderEnd);
    if (0 == commentHeader.compare(0, commentHeader.size(), p, commentHeader.size()))
        return ReadToEndTag(p, commentHeaderEnd);
    if (0 == cdataHeader.compare(0, cdataHeader.size(), p, cdataHeader.size()))
        return ReadToEndTag(p, cdataHeaderEnd);
    if (0 == dtdHeader.compare(0, dtdHeader.size(), p, dtdHeader.size()))
        return ReadToEndTag(p, dtdHeaderEnd);

    return p;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
inline static const char *ReadType(const char *p, std::string &_type)
{
    if (!p || !*p)
        return 0;

    if (!iswalpha(*p) && '_' != *p)
        return 0;

    std::string type = "";
    while (p && *p &&
           (iswalnum(*p) ||
            *p == '_' ||
            *p == '-' ||
            *p == '.' ||
            *p == ':'))
    {
        type += *p;
        ++p;
    }
    _type = type;

    return p;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
const char *RemoveNonXml(const char *p)
{
    p = SkipWhiteSpace(p);
    if (!p || !*p)
        return 0;

    if (*p != '<')
        return 0;

    for (const char *pp = p; pp != (p = ReadNonXml(pp)); pp = SkipWhiteSpace(p))
        ;

    return p;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
const char *Node::AttsFromStr(const char *p)
{
    m_attributes.clear();

    p = SkipWhiteSpace(p);
    char endTag;
    std::string type;
    std::string value;
    while (p && *p && *p != '/' && *p != '>')
    {
        p = SkipWhiteSpace(p);
        if (!p || !*p)
            return 0;

        p = ReadType(p, type);
        if (!p || !*p)
            return 0;

        p = SkipWhiteSpace(p);
        if (!p || !*p || *p != '=')
            return 0;

        ++p;
        p = SkipWhiteSpace(p);
        if (!p || !*p)
            return 0;

        if (*p != '"' && *p != '\'')
            return 0;

        endTag = *p++;

        p = ReadText(p, value, endTag);
        if (!p || !*p)
            return 0;

        m_attributes[type] = value;
        p = SkipWhiteSpace(p);
    }

    return p;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
const char *Node::FromStr(const char *p)
{
    Clear();
    p = ReadNonXml(p);
    if (!p || !*p)
        return 0;

    p = SkipWhiteSpace(p + 1);
    p = ReadType(p, m_type);
    if (!p || !*p)
        return 0;

    p = AttsFromStr(p);
    if (!p || !*p)
        return 0;

    if (*p == '/')
    {
        ++p;
        if (*p != '>')
            return 0;
        return (p + 1);
    }

    if (*p != '>')
        return 0;
    ++p;
    p = SkipWhiteSpace(p);
    if (!p || !*p)
        return 0;

    if (*p != '<')
    {
        const char end = '<';
        p = ReadText(p, m_value, end);
        if (p)
            --p;
    }

    std::string endTag("</");
    endTag += m_type;
    endTag += ">";

    auto me = shared_from_this();
    for (p = RemoveNonXml(p); false == XStrEqual(p, endTag.c_str()); p = RemoveNonXml(p))
    {
        NodePtr node = std::make_shared<Node>("", me);
        p = node->FromStr(p);
        if (0 == p)
            return 0;

        m_children.push_back(node);
    }
    p += endTag.length();
    return p;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(Xml);