@echo off
REM Build the eneural_net CUDA backend on Windows into build\.
REM Requires the NVIDIA CUDA Toolkit (nvcc + cuBLAS) on PATH.

setlocal enabledelayedexpansion

rmdir /s /q build 2>nul
mkdir build

set ARCH=%PROCESSOR_ARCHITECTURE%
if /I "%ARCH%"=="AMD64" (
    set ARCH=x86_64
) else if /I "%ARCH%"=="ARM64" (
    set ARCH=arm64
) else (
    echo Unsupported architecture: %ARCH%
    exit /b 1
)

nvcc -O3 --shared --use_fast_math -lcublas ^
    -gencode arch=compute_70,code=sm_70 ^
    -gencode arch=compute_75,code=sm_75 ^
    -gencode arch=compute_80,code=sm_80 ^
    -gencode arch=compute_86,code=sm_86 ^
    -gencode arch=compute_89,code=sm_89 ^
    -gencode arch=compute_90,code=sm_90 ^
    -I. ^
    -o "build\libeneural_cuda_!ARCH!.dll" ^
    EnnCudaBackend.cu

if %ERRORLEVEL% neq 0 (
    echo Build failed.
    exit /b %ERRORLEVEL%
)

echo Build succeeded: build\libeneural_cuda_!ARCH!.dll
