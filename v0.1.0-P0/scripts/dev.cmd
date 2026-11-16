@echo off
rem P0 dev launcher: SIDE_PYTHON pinned to the 3.14.8 dev venv (D10).
set "SIDE_PYTHON=%~dp0..\.venv\Scripts\python.exe"
cd /d "%~dp0..\src-tauri"
cargo tauri dev
