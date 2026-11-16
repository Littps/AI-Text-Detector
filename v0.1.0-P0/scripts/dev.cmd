@echo off
rem P0 dev launcher: SIDE_PYTHON pinned to the 3.14.8 dev venv.
rem %%~fi canonicalizes the path (no ".." segments) so python's venv
rem detection does not emit sys.prefix RuntimeWarnings.
for %%i in ("%~dp0..\.venv\Scripts\python.exe") do set "SIDE_PYTHON=%%~fi"
cd /d "%~dp0..\src-tauri"
cargo tauri dev
