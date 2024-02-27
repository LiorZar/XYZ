#include "gl/Wnd.h"

int main()
{
    gl::Wnd window(800, 600, "GLFW Hello World");
    auto t = window.Run();
    t.join();

    return 0;
}
