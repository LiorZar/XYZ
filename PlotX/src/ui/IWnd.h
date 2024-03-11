#include "defines.h"
#include "Xml.h"
#include "Factory.hpp"
#include "gui.h"

NAMESPACE_BEGIN(ui);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
class IWnd;
class Node;
class Container;
using NodePtr = std::shared_ptr<Node>;
using ContainerPtr = std::shared_ptr<Container>;
using WeakContainerPtr = std::weak_ptr<Container>;
//-------------------------------------------------------------------------------------------------------------------------------------------------//
using CreateFn = std::function<NodePtr(IWnd *, ContainerPtr, const Xml::Node &)>;
using NodeFactory = Factory<NodePtr, CreateFn>;
//-------------------------------------------------------------------------------------------------------------------------------------------------//
template <typename T>
NodePtr CreateNode(IWnd *wnd, ContainerPtr parent, const Xml::Node &data)
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
class IWnd
{
};
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(ui);