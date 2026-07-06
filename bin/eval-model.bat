@echo off
rem Launcher for scripts\eval-model.py
rem Clears VIRTUAL_ENV so an active conda/miniforge environment is ignored
rem and the project's own uv .venv environment is always used.

setlocal
set "VIRTUAL_ENV="
set "PROJECT_ROOT=%~dp0.."

pushd "%PROJECT_ROOT%" || exit /b 1
uv run python scripts\eval-model.py %*
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" (
    echo.
    echo eval-model exited with an error ^(code %EXIT_CODE%^)
    pause
)

endlocal & exit /b %EXIT_CODE%
