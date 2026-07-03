@echo off
rem Share the data directory over HTTP.
rem Usage: bin\share-data.bat [port] [bind-address]
rem Defaults: port=8000, bind-address=0.0.0.0
rem Clears VIRTUAL_ENV so an active conda/miniforge environment is ignored
rem and the project's own uv .venv environment is always used.

setlocal
set "VIRTUAL_ENV="
set "PROJECT_ROOT=%~dp0.."
set "PORT=%~1"
set "BIND_ADDRESS=%~2"

if "%PORT%"=="" set "PORT=9000"
if "%BIND_ADDRESS%"=="" set "BIND_ADDRESS=0.0.0.0"

pushd "%PROJECT_ROOT%" || exit /b 1

if not exist "data\" (
    echo data directory not found: "%PROJECT_ROOT%\data"
    popd
    endlocal & exit /b 1
)

echo Sharing "%PROJECT_ROOT%\data" over HTTP.
echo.
echo Local:   http://127.0.0.1:%PORT%/
echo Network: http://^<this-computer-ip^>:%PORT%/
echo Bind:    %BIND_ADDRESS%
echo.
echo Press Ctrl+C to stop.
echo.

uv run python -m http.server "%PORT%" --bind "%BIND_ADDRESS%" --directory "data"
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" (
    echo.
    echo share-data exited with an error ^(code %EXIT_CODE%^)
    pause
)

endlocal & exit /b %EXIT_CODE%
