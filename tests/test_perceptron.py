"""Tests for the Perceptron class."""

from perceptron import Perceptron, load_iris_points


def test_initialization() -> None:
    """Проверяем, что при создании перцептрона веса генерируются правильной длины"""
    # Создаем перцептрон для 3 признаков
    my_net = Perceptron(n_features=3)

    # Утверждаем, что в списке весов ровно 3 элемента
    assert len(my_net.wight) == 3
    # Утверждаем, что стартовый bias равен 0
    assert my_net.bias == 0.0


def test_activate_function() -> None:
    """Проверяем, что пороговая функция работает корректно"""
    my_net = Perceptron(n_features=2)

    # Утверждаем, что для нуля и положительных чисел функция вернет 1
    assert my_net.activate(0.0) == 1
    assert my_net.activate(5.5) == 1

    # Утверждаем, что для отрицательных чисел функция вернет 0
    assert my_net.activate(-1.0) == 0
    assert my_net.activate(-100.5) == 0


def test_predict() -> None:
    """Проверяем предсказание на очевидных данных."""
    my_net = Perceptron(n_features=2)
    my_net.wight = [1.0, 0.0]
    my_net.bias = -5.0

    assert my_net.predict([10.0, 0.0]) == 1

    assert my_net.predict([2.0, 0.0]) == 0


def test_train_perfect_separation() -> None:
    """Проверяем, что алгоритм реально умеет учиться и достигает точности 1.0 на простых данных."""
    my_net = Perceptron(n_features=2, learning_rate=0.1, epochs=20)
    X = [
        [0.0, 1.0],
        [0.0, 2.0],  # Класс 0
        [10.0, 1.0],
        [10.0, 2.0],  # Класс 1
    ]
    y = [0, 0, 1, 1]
    my_net.train(X, y)
    assert my_net.score(X, y) == 1.0
    assert len(my_net.list_error) == 20


def test_train_on_iris() -> None:
    """Проверка на реальном наборе данных iris (setosa против остальных)."""
    X, y = load_iris_points()
    my_net = Perceptron(n_features=2, learning_rate=0.1, epochs=50)
    my_net.train(X, y)
    assert my_net.score(X, y) >= 0.99
