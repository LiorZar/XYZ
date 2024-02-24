rmdir /Q /S cudalink
mkdir cudalink

mklink /D "cudalink\bin" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
mklink /D "cudalink\lib" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib"
mklink /D "cudalink\include" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include"
mklink /D "cudalink\common" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\samples\Common"




 
