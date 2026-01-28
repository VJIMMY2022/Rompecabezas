@echo off
echo ==========================================
echo    PREPARANDO PROYECTO PARA GITHUB
echo ==========================================
echo.
echo 1. Inicializando repositorio...
git init
git branch -M main

echo.
echo 2. Agregando archivos...
git add .

echo.
echo 3. Creando commit inicial...
git commit -m "Primera version - Puzzle Solver AI"

echo.
echo ==========================================
echo    CONECTANDO CON GITHUB
echo ==========================================
echo.
echo Por favor, ve a https://github.com/new y crea un nuevo repositorio.
echo (No marques ninguna casilla de README, .gitignore o licencia)
echo.
set /p repo_url="Pega aqui el link del repositorio (ej: https://github.com/tu_usuario/mi_repo.git): "

echo.
echo 4. Vinculando repositorio remoto...
git remote add origin %repo_url%

echo.
echo 5. Subiendo archivos (esto puede pedirte tu usuario/clave)...
git push -u origin main

echo.
echo ==========================================
echo             ¡LISTO!
echo ==========================================
echo Ahora ve a https://share.streamlit.io/
echo Conecta tu GitHub y selecciona el repositorio: %repo_url%
echo.
pause
