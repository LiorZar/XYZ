#include "cu/GPU.h"
#include "SP.cuh"

using namespace cu;
int main()
{
    std::cout << "Current Dir: " << GPU::GetCurrentDirectory() << std::endl;
    std::cout << "Working Dir: " << GPU::GetWorkingDirectory() << std::endl;

    GPU::Set();
    SP sp;

//#define SUPERMAN
#ifdef SUPERMAN
    if (false == sp.SupermanInit())
        return 1;

    if (false == sp.SupermanProcess())
        return 2;
#endif

#define SPIDERMAN
#ifdef SPIDERMAN
    if (false == sp.SpidermanInit())
        return 1;

#define VER 3
    for (int channel = 1; channel <= 1; ++channel)
    {
        if (false == sp.SpidermanLoad(channel))
            continue;

#if VER == 1
           for(int i = 1; i <= 24; ++i)
               sp.SpidermanSingleProcess(i);
#elif VER == 2
           for(int i = 1; i <= 24; ++i)
               sp.SpidermanSingleSplitProcess(i);
#elif VER == 3
		if(false == sp.SpidermanBatchSplitProcess())
			return 2;        
#elif VER == 4
		if(false == sp.SpidermanBatchProcess())
			return 2;
#endif
        std::cout << "channel " << channel << " success\n";
#endif
    }

    return 0;
}
