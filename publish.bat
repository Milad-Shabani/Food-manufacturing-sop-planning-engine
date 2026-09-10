@echo off
REM ============================================================
REM Publish this folder as a new public repo on github.com/Milad-Shabani
REM Requires: git, and GitHub CLI (gh) installed + logged in (gh auth login)
REM ============================================================

set REPO_NAME=food-manufacturing-sop-planning-engine
set "REPO_DESC=S&OP planning engine for a 5-line industrial cake & confectionery food manufacturer: demand forecasting (Holt-Winters), capacity- and labor-constrained production planning (LP), MRP, and a simulated cash conversion cycle (DIO/DSO/DPO), on a realistic synthetic dataset."

REM --- adjust this to wherever you unzipped/cloned the project locally
cd /d "C:\Users\MILAD\Desktop\food-manufacturing-sop-planning-engine"

REM --- set your git identity (safe to run every time)
git config --global user.name "Milad Shabani"
git config --global user.email "MILAD.SHABANI6515@GMAIL.COM"

REM --- init only if not already a repo
if not exist ".git" (
    git init
    git branch -M main
)

REM --- remove any leftover remote from a previous attempt
git remote remove origin 2>nul

git add .
git commit -m "Initial commit: Food Manufacturing S&OP Planning Engine"
git branch -M main

gh repo create %REPO_NAME% --public --source=. --remote=origin --push --description "%REPO_DESC%"

gh repo edit Milad-Shabani/%REPO_NAME% --add-topic supply-chain --add-topic demand-forecasting --add-topic production-planning --add-topic operations-research --add-topic linear-programming --add-topic python --add-topic power-bi

echo.
echo Done. Repo should now be live at:
echo https://github.com/Milad-Shabani/%REPO_NAME%
pause
