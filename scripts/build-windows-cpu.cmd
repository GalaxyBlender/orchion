@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0.."

set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"

if not defined VS_VCVARS64 (
    for /f "delims=" %%I in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "$vs = & '%VSWHERE%' -latest -prerelease -format json -property installationPath | ConvertFrom-Json; if ($vs -is [array]) { $vs[0].installationPath } else { $vs.installationPath }"') do set "VS_INSTALLATION_PATH=%%I"
    if not defined VS_INSTALLATION_PATH (
        echo Failed to detect Visual Studio installation with vswhere.
        exit /b 1
    )
    set "VS_VCVARS64=!VS_INSTALLATION_PATH!\VC\Auxiliary\Build\vcvars64.bat"
)

call "%VS_VCVARS64%"
if errorlevel 1 exit /b %errorlevel%

set "CFLAGS=/MD"
set "CXXFLAGS=/MD"

cargo build --release -p orchion-server --no-default-features --features cpu
