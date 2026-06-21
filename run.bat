@echo off
cd /d "%~dp0"

:: HF_TOKEN 환경 변수가 설정되어 있지 않다면 입력을 받습니다.
if not "%HF_TOKEN%"=="" goto ACTIVATE

echo.
echo =======================================================================
echo [안내] Gemma 3 모델은 Hugging Face 약관 동의 및 토큰 인증이 필요합니다.
echo 1. https://huggingface.co/google/gemma-3-1b-pt 에서 동의를 해야 합니다.
echo 2. https://huggingface.co/settings/tokens 에서 토큰 [Read 권한] 을 복사해 입력하세요.
echo =======================================================================
echo.
set /p HF_TOKEN="HF_TOKEN 입력 (엔터 시 건너뜀): "

:ACTIVATE
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Running gomi.py...
python gomi.py
pause


