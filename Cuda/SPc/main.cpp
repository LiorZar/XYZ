#include "cu/GPU.h"
#include "SP.cuh"

using namespace cu;
int main()
{
    std::cout << "Current Dir: " << GPU::GetCurrentDirectory() << std::endl;
    std::cout << "Working Dir: " << GPU::GetWorkingDirectory() << std::endl;

    GPU::Set();
    {
        SP sp;
        if (false == sp.Init())
            return 1;
        //        if (false == sp.XYZProcess())
        //            return 2;
        for (int i = 1; i <= 24; ++i)
            sp.SingleZYXProcess(i);

        //        for(int i = 1; i <= 1; ++i)
        //            sp.Decimate(i);
        //        for(int i = 1; i <= 24; ++i)
        //            sp.Chrip(i);
        //
        //        for(int i = 1; i <= 1; ++i)
        //            sp.STFT(i);

        //        if (false == sp.STFT())
        //            return 3;
        //        if (false == sp.MinMax())
        //            return 4;
    }

    return 0;
}
