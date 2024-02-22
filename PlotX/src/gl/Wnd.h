#ifndef GLFW_WINDOW_WRAPPER_H
#define GLFW_WINDOW_WRAPPER_H

#include "defines.h"

class Wnd
{
private:
    GLFWwindow *window;
    int width, height;

public:
    Wnd(int width, int height, const char *title);
    ~Wnd();

    bool Loop();

protected:
    virtual bool DrawScene();
    virtual void OnMouse(int button, int action, int mods);
    virtual void OnKey(int key, int scancode, int action, int mods);
    virtual void OnWindowClose();
    virtual void OnWindowSize(int width, int height);
    virtual void OnWindowRefresh();
    virtual void OnCursorPosition(double xpos, double ypos);
    virtual void OnScroll(double xoffset, double yoffset);
    virtual void OnChar(unsigned int codepoint);

private:
    bool shouldClose() const;
    void swapBuffers();
    void pollEvents();
    GLFWwindow *get() const;
};

#endif // GLFW_WINDOW_WRAPPER_H