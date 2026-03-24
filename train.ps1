# Скрипт для запуска обучения на Windows PowerShell
# Убедитесь, что venv активирован!

Write-Host "Проверка виртуального окружения..." -ForegroundColor Cyan
try {
    $pythonPath = python -c "import sys; print(sys.executable)" 2>&1
    Write-Host "Python: $pythonPath" -ForegroundColor Green
} catch {
    Write-Host "ОШИБКА: Виртуальное окружение не активировано!" -ForegroundColor Red
    Write-Host "Запустите: .\venv\Scripts\activate.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "Проверка torch..." -ForegroundColor Cyan
try {
    $torchVersion = python -c "import torch; print(torch.__version__)" 2>&1
    Write-Host "Torch OK: $torchVersion" -ForegroundColor Green
} catch {
    Write-Host "ОШИБКА: torch не установлен!" -ForegroundColor Red
    Write-Host "Установите: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu" -ForegroundColor Yellow
    exit 1
}

Write-Host "Запуск обучения..." -ForegroundColor Cyan
python train_selfplay.py --total-steps 200000 --save-dir runs/ppo --visual-monitor --visual-freq 10

