@echo off
title Iklim-Eval Proje Kurulumu
echo ======================================================
echo          IKLIM-EVAL PROJE KURULUMU STARTING...
echo ======================================================
echo.

:: 1. Python kontrolü
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [HATA] Python bulunamadi! Lutfen Python yukleyin.
    pause
    exit /b 1
)

:: 2. Sanal ortam (venv) olusturma
if not exist "%~dp0venv" (
    echo [1/3] Sanal ortam olusturuluyor...
    python -m venv "%~dp0venv"
) else (
    echo [1/3] venv zaten mevcut.
)

:: 3. Bagimliliklarin kurulumu
echo [2/3] Bagimliliklar yukleniyor...
"%~dp0venv\Scripts\python.exe" -m pip install --upgrade pip
"%~dp0venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if %errorlevel% neq 0 (
    echo [HATA] Kutuphaneler yuklenirken bir sorun olustu!
    pause
    exit /b 1
)

:: 4. .env dosyasi kontrolu
if not exist .env (
    echo ANTHROPIC_API_KEY=sk-ant-buraya-yaziniz > .env
    echo ANTHROPIC_MODEL=claude-3-haiku-20240307 >> .env
    echo [3/3] .env dosyasi olusturuldu.
) else (
    echo [3/3] .env mevcut.
)

echo.
echo ======================================================
echo    KURULUM TAMAMLANDI!
echo ======================================================
echo.
pause
