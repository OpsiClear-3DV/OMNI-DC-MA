@echo off
REM Build the DCN CUDA extension for OMNI-DC against torch 2.11 + CUDA 13.0.
REM Sets up the MSVC env and points NVCC at the 5090 (sm_120 / Blackwell).

setlocal ENABLEDELAYEDEXPANSION
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "CUDA_PATH=%CUDA_HOME%"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "DISTUTILS_USE_SDK=1"

call "%VCVARS%" || exit /b 1

set "REPO=%~dp0.."
cd /d "%REPO%\src\model\deformconv" || exit /b 1

REM Use the uv-managed Python so the build links against torch 2.11 + cu130.
"%REPO%\.venv\Scripts\python.exe" setup.py build_ext --inplace
exit /b %ERRORLEVEL%
