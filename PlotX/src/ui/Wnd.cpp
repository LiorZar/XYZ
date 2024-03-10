#include "Wnd.h"

NAMESPACE_BEGIN(ui);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
Wnd::Wnd(int _width, int _height, const char *_title) : width(_width), height(_height), title(_title)
{
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
Wnd::~Wnd()
{
    glfwDestroyWindow(window);
    glfwTerminate();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
std::thread Wnd::Run()
{
    return std::thread([this]
                       { Loop(); });
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Wnd::Loop()
{
    if (!glfwInit())
        throw std::runtime_error("Failed to initialize GLFW");

    window = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, this);

    glfwSetMouseButtonCallback(
        window,
        [](GLFWwindow *window, int button, int action, int mods)
        {
            Wnd *wnd = static_cast<Wnd *>(glfwGetWindowUserPointer(window));
            wnd->OnMouse(button, action, mods);
        });
    glfwSetKeyCallback(
        window,
        [](GLFWwindow *window, int key, int scancode, int action, int mods)
        {
            Wnd *wnd = static_cast<Wnd *>(glfwGetWindowUserPointer(window));
            wnd->OnKey(key, scancode, action, mods);
        });
    glfwSetWindowCloseCallback(
        window,
        [](GLFWwindow *window)
        {
            Wnd *wnd = static_cast<Wnd *>(glfwGetWindowUserPointer(window));
            wnd->OnWindowClose();
        });
    glfwSetWindowSizeCallback(
        window,
        [](GLFWwindow *window, int width, int height)
        {
            Wnd *wnd = static_cast<Wnd *>(glfwGetWindowUserPointer(window));
            wnd->OnWindowSize(width, height);
        });
    glfwSetWindowRefreshCallback(
        window,
        [](GLFWwindow *window)
        {
            Wnd *wnd = static_cast<Wnd *>(glfwGetWindowUserPointer(window));
            wnd->OnWindowRefresh();
        });
    glfwSetCursorPosCallback(
        window,
        [](GLFWwindow *window, double xpos, double ypos)
        {
            Wnd *wnd = static_cast<Wnd *>(glfwGetWindowUserPointer(window));
            wnd->OnCursorPosition(xpos, ypos);
        });
    glfwSetScrollCallback(
        window,
        [](GLFWwindow *window, double xoffset, double yoffset)
        {
            Wnd *wnd = static_cast<Wnd *>(glfwGetWindowUserPointer(window));
            wnd->OnScroll(xoffset, yoffset);
        });
    glfwSetCharCallback(
        window,
        [](GLFWwindow *window, unsigned int codepoint)
        {
            Wnd *wnd = static_cast<Wnd *>(glfwGetWindowUserPointer(window));
            wnd->OnChar(codepoint);
        });

    while (!shouldClose())
    {
        if (!DrawScene())
            break;
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Wnd::DrawScene()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_TRIANGLES);
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex2f(-0.5f, -0.5f);
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex2f(0.5f, -0.5f);
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex2f(0.0f, 0.5f);
    glEnd();
    swapBuffers();
    pollEvents();

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Wnd::OnMouse(int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        std::cout << "Left mouse button pressed at position (" << xpos << ", " << ypos << ")" << std::endl;
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Wnd::OnKey(int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Wnd::OnWindowClose()
{
    std::cout << "Window is closing" << std::endl;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Wnd::OnWindowSize(int width, int height)
{
    std::cout << "Window size changed to " << width << "x" << height << std::endl;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Wnd::OnWindowRefresh()
{
    std::cout << "Window content needs to be refreshed" << std::endl;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Wnd::OnCursorPosition(double xpos, double ypos)
{
    std::cout << "Cursor position: (" << xpos << ", " << ypos << ")" << std::endl;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Wnd::OnScroll(double xoffset, double yoffset)
{
    std::cout << "Scroll offset: (" << xoffset << ", " << yoffset << ")" << std::endl;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Wnd::OnChar(unsigned int codepoint)
{
    std::cout << "Character typed: " << static_cast<char>(codepoint) << std::endl;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Wnd::shouldClose() const
{
    return glfwWindowShouldClose(window);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Wnd::swapBuffers()
{
    glfwSwapBuffers(window);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Wnd::pollEvents()
{
    glfwPollEvents();
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
GLFWwindow *Wnd::get() const
{
    return window;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(ui);