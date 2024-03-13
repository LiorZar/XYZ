#include "defines.h"
#include "Xml.h"
#include "Factory.hpp"
#include "gui.h"

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
ControlPtr CreateControl(IWnd *wnd, ContainerPtr parent, const Xml::Node &data);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
class IWnd
{
public:
    IWnd() = default;
    virtual ~IWnd() = default;

public:
    virtual Xml::NodePtr GetLibraryNode(Xml::NodePtr node) const;
};
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(ui);