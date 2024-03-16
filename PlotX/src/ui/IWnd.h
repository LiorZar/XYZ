#include "defines.h"
#include "Xml.h"
#include "Factory.hpp"
#include "gui.h"
#include "gl/ResourceManager.h"

NAMESPACE_BEGIN(ui);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
class IWnd;
class Control;
class Container;
using ControlPtr = std::shared_ptr<Control>;
using ContainerPtr = std::shared_ptr<Container>;
using WeakContainerPtr = std::weak_ptr<Container>;
using ControlFn = std::function<void(ControlPtr)>;
//-------------------------------------------------------------------------------------------------------------------------------------------------//
using CreateFn = std::function<ControlPtr(IWnd *, ContainerPtr, const Xml::Node &)>;
using NodeFactory = Factory<CreateFn>;
//-------------------------------------------------------------------------------------------------------------------------------------------------//
template <typename T>
ControlPtr CreateNode(IWnd *wnd, ContainerPtr parent, const Xml::Node &data)
{
    auto node = std::make_shared<T>(wnd, parent);
    if (nullptr == node)
        return nullptr;

    if (false == node->Parse(data))
        return nullptr;

    if (false == node->PostInit())
        return nullptr;

    return node;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
template <typename T>
void RegisterNode(const std::string &name)
{
    NodeFactory::Register(name, CreateNode<T>);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
template <typename T>
void RegisterNode(const std::vector<std::string> &names)
{
    auto fn = CreateNode<T>;
    for (const auto &name : names)
    {
        NodeFactory::Register(name, fn);
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
#define REGISTER_NODE(name)                               \
    struct name##staticInit                               \
    {                                                     \
        name##staticInit() { RegisterNode<name>(#name); } \
    };                                                    \
    static name##staticInit s_name##staticInit;
//-------------------------------------------------------------------------------------------------------------------------------------------------//
#define REGISTER_NODE_NAMES(name, ...)                            \
    struct name##staticInit                                       \
    {                                                             \
        name##staticInit() { RegisterNode<name>({__VA_ARGS__}); } \
    };                                                            \
    static name##staticInit s_name##staticInit;
//-------------------------------------------------------------------------------------------------------------------------------------------------//
ControlPtr CreateControl(IWnd *wnd, ContainerPtr parent, const Xml::Node &data);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
class IWnd
{
public:
    IWnd(const std::string &workDir);
    virtual ~IWnd() = default;

public:
    virtual void InitGL();

public:
    virtual Xml::NodePtr GetLibraryNode(Xml::NodePtr node) const;

protected:
    std::string m_workDir;
    Xml::NodePtr m_root;
    int m_width = 800, m_height = 600;

protected:
    gl::ResourceManeger m_resources;
};
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(ui);