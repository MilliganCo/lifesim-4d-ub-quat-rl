@echo off
REM Скрипт для запуска обучения на Windows
REM Убедитесь, что venv активирован!

echo Проверка виртуального окружения...
python -c "import sys; print('Python:', sys.executable)" || (
    echo ОШИБКА: Виртуальное окружение не активировано!
    echo Запустите: venv\Scripts\activate
    pause
    exit /b 1
)

echo Проверка torch...
python -c "import torch; print('Torch OK:', torch.__version__)" || (
    echo ОШИБКА: torch не установлен!
    echo Установите: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pause
    exit /b 1
)

echo Запуск обучения...
python train_selfplay.py --total-steps 200000 --save-dir runs/ppo --visual-monitor --visual-freq 10

pause

