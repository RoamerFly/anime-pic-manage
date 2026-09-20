@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist "anime-pic-manage.exe" (
    echo [错误] 没有找到 anime-pic-manage.exe，请确认压缩包已完整解压。
    pause
    exit /b 1
)
if not exist "app\runtime\ai-worker.exe" (
    echo [错误] 缺少 app\runtime\ai-worker.exe，压缩包可能没有解压完整。
    echo        请删除后重新解压整个文件夹，不要只解压单个文件。
    pause
    exit /b 1
)
start "" "anime-pic-manage.exe"
exit /b 0
