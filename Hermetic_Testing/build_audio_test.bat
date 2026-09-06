@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cl /std:c++20 /EHsc /O2 /I. HermeticAudioTest.cpp /Fe:HermeticAudioTest.exe
