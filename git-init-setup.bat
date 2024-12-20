@echo off
REM 替换为你的默认远程仓库地址
set REPO_URL=https://github.com/66681/fasterrcnn.git

REM 初始化 Git 仓库
git init

REM 添加默认远程仓库地址
git remote add origin %REPO_URL%

echo Git repository initialized and remote set to %REPO_URL%.
