#include "ui/Wnd.h"

int main()
{
    std::filesystem::path workDir = std::filesystem::current_path();
    auto path = workDir / "../assets/";

    auto window = std::make_shared<ui::Wnd>(path.string());
    if (!window->Load("layout.xml"))
        return -1;

    auto t = window->Start();
    t.join();

    return 0;
}
