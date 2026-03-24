# LifeSim 4-Team with 4D-UV-Quat RL

Минимальная симуляция «выживания» для 4 команд на сетке.
Одна команда обучается через самоигру (self-play) с системой вознаграждений.
Политика использует свёрточный экстрактор признаков с идеей 4D-UV-Quat (сдвиги U,V и «кватернионное» смешивание).

## Новые возможности ✨

- **Визуальный клиент** на pygame для просмотра игры в реальном времени
- **Система сохранения реплеев** для анализа эпизодов
- **Визуальный мониторинг** во время обучения
- Улучшенные метрики и статистика

## Быстрый старт

### Windows

```powershell
# Создание виртуального окружения
python -m venv venv

# Активация (PowerShell)
.\venv\Scripts\activate.ps1

# Если возникает ошибка выполнения скриптов, выполните:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Установка зависимостей
pip install -r requirements.txt

# Установка torch для Windows (важно!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Linux/Mac

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch>=2.1
```

### Обучение self-play

**Важно:** Убедитесь, что виртуальное окружение активировано!

**Windows (PowerShell):**
```powershell
.\venv\Scripts\activate.ps1
python train_selfplay.py --total-steps 200000 --save-dir runs/ppo --visual-monitor --visual-freq 10
```

**Или используйте готовый скрипт:**
```powershell
.\train.ps1
```

**Базовое обучение:**
```bash
python train_selfplay.py --total-steps 200000 --save-dir runs/ppo
```

**С визуальным просмотром во время обучения:**
```bash
python train_selfplay.py --total-steps 200000 --save-dir runs/ppo --visual-monitor --visual-freq 10
```

### Просмотр игры с моделью

**Визуальный режим (рекомендуется):**
```bash
python play.py --ckpt runs/ppo/best_model.zip --episodes 3 --render visual
```

**ASCII режим:**
```bash
python play.py --ckpt runs/ppo/best_model.zip --episodes 3 --render ascii
```

**С сохранением реплеев:**
```bash
python play.py --ckpt runs/ppo/best_model.zip --episodes 3 --save-replay --replay-dir replays
```

## Идея 4D-UV-Quat в политике

Экстрактор переводит карту [C,H,W] в «4D поле» через разбиение на U,V=2.
Затем применяет QuaternionUVConv1x1: линейное смешивание четырёх фазовых сдвигов (0, U, V, U+V).
Это повышает трансляционную устойчивость и учит единый фильтр работать для разных выравниваний.

## Визуализация

Визуальный клиент показывает:
- **Ресурсы**: Зеленый (еда), Коричневый (дерево), Серый (руда)
- **Здания**: Рынки (M), Склады (D), Фермы (F), Казармы (B)
- **Юниты**: Рабочие (W), Атакующие (A), Защитники (D), Разведчики (S)
- **Команды**: Разные цвета для каждой команды (0-3)
- **Обучающаяся команда**: Подсвечена желтой рамкой

## Параметры командной строки

### train_selfplay.py
- `--total-steps`: Общее количество шагов обучения (по умолчанию: 200000)
- `--save-dir`: Директория для сохранения моделей (по умолчанию: runs/ppo)
- `--visual-monitor`: Включить визуальный мониторинг во время обучения
- `--visual-freq`: Частота показа эпизодов (каждые N чекпоинтов, по умолчанию: 10)
- `--learn-team`: Команда для обучения (-1 = случайная, по умолчанию: -1)

### play.py
- `--ckpt`: Путь к чекпоинту модели (обязательно)
- `--episodes`: Количество эпизодов для просмотра (по умолчанию: 2)
- `--render`: Режим рендеринга: `visual` или `ascii` (по умолчанию: visual)
- `--team`: Команда для управления (-1 = случайная, по умолчанию: -1)
- `--save-replay`: Сохранять реплеи эпизодов
- `--replay-dir`: Директория для сохранения реплеев (по умолчанию: replays)

## Структура проекта

- `engine.py` - Игровая логика и симуляция
- `env.py` - Gymnasium окружение с поддержкой визуализации и реплеев
- `models.py` - 4D-UV-Quat экстрактор признаков
- `train_selfplay.py` - Обучение с self-play
- `play.py` - Просмотр обученных моделей
- `visualizer.py` - Визуальный клиент на pygame

## Ограничения демо
- Простые правила боя и добычи.
- Торговля по фиксированным ценам на клетках рынка.
- Один обучаемый игрок, соперники — последняя «заморозка» или простые боты.
