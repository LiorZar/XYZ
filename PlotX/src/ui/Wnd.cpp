#include "Wnd.h"

NAMESPACE_BEGIN(ui);
//-------------------------------------------------------------------------------------------------------------------------------------------------//
Wnd::Wnd(const std::string &workDir) : Container(this, nullptr), IWnd(workDir)
{
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
Wnd::~Wnd()
{
    if (nullptr != window)
    {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Wnd::Load(const std::string &filename)
{
    m_root = Xml::FromFile(m_workDir + filename);
    if (nullptr == m_root)
        return false;

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
std::thread Wnd::Start()
{
    return std::thread([this]
                       { Run(); });
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Wnd::Run()
{
    if (nullptr == m_root)
        throw std::runtime_error("No layout");

    if (!glfwInit())
        throw std::runtime_error("Failed to initialize GLFW");

    window = glfwCreateWindow(m_width, m_height, m_root->attr("title", "no title").c_str(), NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, this);

    if (GLEW_OK != glewInit())
        throw std::runtime_error("Failed to initialize GLEW");

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

    InitGL();
    OnWindowSize(m_width, m_height);
    while (!glfwWindowShouldClose(window))
    {
        if (false == Update())
            break;
        if (false == PreScene() || false == DrawScene() || false == PostScene())
            break;
    }
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Wnd::Update()
{
    glfwPollEvents();

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Wnd::PreScene()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Wnd::DrawScene()
{
    static float L = 50.f, H = 500.f;
    glBegin(GL_QUADS);
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex2f(L, L);

    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex2f(H, L);

    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex2f(H, H);

    glColor3f(1.0f, 1.0f, 1.0f);
    glVertex2f(L, H);
    glEnd();

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Wnd::PostScene()
{
    glfwSwapBuffers(window);

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
bool Wnd::OnKey(int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    return true;
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
    m_width = width;
    m_height = height;

    glViewport(0, 0, m_width, m_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, m_width, m_height, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Wnd::OnWindowRefresh()
{
    std::cout << "Window content needs to be refreshed" << std::endl;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Wnd::OnCursorPosition(double xpos, double ypos)
{
    // std::cout << "Cursor position: (" << xpos << ", " << ypos << ")" << std::endl;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
void Wnd::OnScroll(double xoffset, double yoffset)
{
    // std::cout << "Scroll offset: (" << xoffset << ", " << yoffset << ")" << std::endl;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
bool Wnd::OnChar(unsigned int codepoint)
{
    // std::cout << "Character typed: " << static_cast<char>(codepoint) << std::endl;

    return true;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------//
NAMESPACE_END(ui);