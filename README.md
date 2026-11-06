# Лабораторная 1 · Перцептрон

Однослойный перцептрон для бинарной классификации (классы 0/1), написан вручную.
Данные: синтетические (2D, два класса вокруг разных центров).
После обучения печатается точность и показываются графики: граница решения
и число ошибок по эпохам. Работа проверяется также на наборе iris (в тестах).

## Установка
    uv sync

## Запуск
    uv run python perceptron.py --samples 200 --lr 0.1 --epochs 50

## Тесты и проверки
    uv run pytest
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy --strict perceptron.py

## Структура
- `perceptron.py` - класс `Perceptron`, генерация данных, графики, CLI
- `tests/test_perceptron.py` - тесты pytest
- `.github/workflows/ci.yml` - CI (ruff, mypy, pytest на каждый push)