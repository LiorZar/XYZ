rmdir /Q /S cudalink
mkdir cudalink

mklink /D "cudalink\bin" "D:\Program Files\Nvidia\bin"
mklink /D "cudalink\lib" "D:\Program Files\Nvidia\lib"
mklink /D "cudalink\include" "D:\Program Files\Nvidia\include"
mklink /D "cudalink\common" "D:\Program Files\Nvidia\samples\Common"




 
