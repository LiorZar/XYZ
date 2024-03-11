#ifndef __XML_H__
#define __XML_H__

#include "defines.h"
#include "vec.h"

NAMESPACE_BEGIN(Xml);

class Node;
using NodePtr = std::shared_ptr<Node>;
using WeakNodePtr = std::weak_ptr<Node>;

class Node : public std::enable_shared_from_this<Node>
{
public:
    Node(const std::string &type = "", NodePtr parent = nullptr, const std::string &text = "");
    Node(const Node &node) = delete;
    NodePtr operator=(const Node &node);
    NodePtr Clone() const;
    void Clear();
    NodePtr Merge(NodePtr node);
    virtual ~Node();

public:
    static NodePtr FromFile(const std::string &path);
    static NodePtr FromString(const std::string &str);

public:
    inline const std::string &Type() const { return m_type; }
    inline void Type(std::string &type) { m_type = type; }
    inline const std::string &Value() const { return m_value; }
    inline void Value(std::string &value) { m_value = value; }
    inline NodePtr Parent() const { return m_parent.lock(); }
    inline size_t ChildrenCount() const { return m_children.size(); }

public:
    bool get(const std::string &name, bul &val, bul dflt = 0) const;
    bool get(const std::string &name, s08 &val, s08 dflt = 0) const;
    bool get(const std::string &name, u08 &val, u08 dflt = 0) const;
    bool get(const std::string &name, s16 &val, s16 dflt = 0) const;
    bool get(const std::string &name, u16 &val, u16 dflt = 0) const;
    bool get(const std::string &name, s32 &val, s32 dflt = 0) const;
    bool get(const std::string &name, u32 &val, u32 dflt = 0) const;
    bool get(const std::string &name, s64 &val, s64 dflt = 0) const;
    bool get(const std::string &name, u64 &val, u64 dflt = 0) const;
    bool get(const std::string &name, f32 &val, f32 dflt = 0) const;
    bool get(const std::string &name, f64 &val, f64 dflt = 0) const;
    bool get(const std::string &name, std::string &val, const std::string &dflt = "") const;
    std::string attr(const std::string &name, const std::string &dflt = "") const;
    template <typename T>
    T Get(const std::string &name, const T &dflt = T()) const
    {
        T val;
        if (get(name, val, dflt))
            return val;
        return dflt;
    }
    vec4 getColor(const std::string &name, const vec4 &dflt = vec4()) const;

public:
    bool get(const std::string &name, vec2 &val, const vec2 &dflt = vec2()) const;
    bool get(const std::string &name, vec3 &val, const vec3 &dflt = vec3()) const;
    bool get(const std::string &name, vec4 &val, const vec4 &dflt = vec4()) const;
    bool get(const std::string &name, dvec2 &val, const dvec2 &dflt = dvec2()) const;
    bool get(const std::string &name, dvec3 &val, const dvec3 &dflt = dvec3()) const;
    bool get(const std::string &name, dvec4 &val, const dvec4 &dflt = dvec4()) const;
    bool get(const std::string &name, ivec2 &val, const ivec2 &dflt = ivec2()) const;
    bool get(const std::string &name, ivec3 &val, const ivec3 &dflt = ivec3()) const;
    bool get(const std::string &name, ivec4 &val, const ivec4 &dflt = ivec4()) const;
    bool get(const std::string &name, uvec2 &val, const uvec2 &dflt = uvec2()) const;
    bool get(const std::string &name, uvec3 &val, const uvec3 &dflt = uvec3()) const;
    bool get(const std::string &name, uvec4 &val, const uvec4 &dflt = uvec4()) const;

public:
    bool get(const std::string &name, std::vector<bul> &vals) const;
    bool get(const std::string &name, std::vector<s08> &vals) const;
    bool get(const std::string &name, std::vector<u08> &vals) const;
    bool get(const std::string &name, std::vector<s16> &vals) const;
    bool get(const std::string &name, std::vector<u16> &vals) const;
    bool get(const std::string &name, std::vector<s32> &vals) const;
    bool get(const std::string &name, std::vector<u32> &vals) const;
    bool get(const std::string &name, std::vector<s64> &vals) const;
    bool get(const std::string &name, std::vector<u64> &vals) const;
    bool get(const std::string &name, std::vector<f32> &vals) const;
    bool get(const std::string &name, std::vector<f64> &vals) const;
    bool get(const std::string &name, std::vector<std::string> &vals) const;

public:
    bool getX(const std::string &name, ivec2 &val) const;
    bool getX(const std::string &name, ivec3 &val) const;
    bool getX(const std::string &name, ivec4 &val) const;
    bool getX(const std::string &name, uvec2 &val) const;
    bool getX(const std::string &name, uvec3 &val) const;
    bool getX(const std::string &name, uvec4 &val) const;
    bool getX(const std::string &name, std::vector<s32> &vals) const;
    bool getX(const std::string &name, std::vector<u32> &vals) const;

public:
    NodePtr First(const std::string &type = "") const;
    NodePtr Last(const std::string &type = "") const;
    NodePtr Prev(const std::string &type = "") const;
    NodePtr Next(const std::string &type = "") const;
    NodePtr ByName(const std::string &name = "") const { return ByAttr("name", name); }
    NodePtr ByAttr(const std::string &att, const std::string &value) const;

public:
    std::vector<NodePtr> Children(const std::string &type = "") const;

public:
    bool InsertEnd(NodePtr addThis);
    bool InsertBefore(NodePtr beforeThis, NodePtr addThis);
    bool InsertAfter(NodePtr afterThis, NodePtr addThis);
    bool Replace(NodePtr replaceThis, NodePtr addThis);
    bool Remove(NodePtr removeThis);
    bool RemoveFromParent();

private:
    const char *FromStr(const char *p);
    const char *AttsFromStr(const char *p);

private:
    std::string m_type;
    std::string m_value;
    WeakNodePtr m_parent;
    std::list<NodePtr> m_children;
    std::unordered_map<std::string, std::string> m_attributes;
};
NAMESPACE_END(Xml);

#endif // __XML_H__