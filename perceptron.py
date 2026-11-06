"""Perceptron: a single-layer neural network for binary classification."""

import argparse

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_blobs


class Perceptron:
    """Single-layer perceptron with a threshold activation (outputs 0 or 1)."""

    def __init__(self, n_features: int, learning_rate: float = 0.1, epochs: int = 10) -> None:
        """Create a perceptron with zero weights and zero bias."""
        self.lr = learning_rate
        self.epochs = epochs
        self.n = n_features
        self.wight: list[float] = [0.0 for x in range(n_features)]
        self.bias: float = 0.0
        self.list_error: list[int] = []

    def activate(self, x: float) -> int:
        """Threshold function: 1 if x >= 0, otherwise 0."""
        if x < 0:
            return 0
        else:
            return 1

    def predict(self, x: list[float]) -> int:
        """Forward step: weighted sum plus bias, then activation."""
        s = self.bias
        for i in range(self.n):
            s += x[i] * self.wight[i]
        answer = self.activate(s)
        return answer

    def train(self, X: list[list[float]], y: list[int]) -> None:
        """Train with the rule w += lr * (target - pred) * x.

        The number of misclassified samples per epoch is saved in list_error.
        """
        self.list_error = []
        for cycle in range(self.epochs):
            cur_error = 0
            for i in range(len(X)):
                current_answer = self.predict(X[i])
                if current_answer != y[i]:
                    cur_error += 1
                    for curr in range(self.n):
                        self.wight[curr] += self.lr * (y[i] - current_answer) * X[i][curr]
                    self.bias += self.lr * (y[i] - current_answer)
            self.list_error.append(cur_error)

    def score(self, X: list[list[float]], y: list[int]) -> float:
        """Return the share of correctly classified samples (0.0 .. 1.0)."""
        count_rifght = 0
        for i in range(len(X)):
            if self.predict(X[i]) == y[i]:
                count_rifght += 1
        return count_rifght / len(X)


def generate_points(num: int) -> tuple[list[list[float]], list[int]]:
    """Generate num linearly separable 2D points of two classes."""
    X_mn, y_ans = make_blobs(n_samples=num, n_features=2, centers=2, random_state=15)
    X_mn = X_mn.tolist()
    y_ans = y_ans.tolist()
    return X_mn, y_ans


def load_iris_points() -> tuple[list[list[float]], list[int]]:
    """Real dataset: iris, class 1 = setosa, class 0 = the others.

    Only petal length and petal width are used, so the classes are separable.
    """
    data = load_iris()
    X_iris = [[float(row[2]), float(row[3])] for row in data.data]
    y_iris = [1 if label == 0 else 0 for label in data.target]
    return X_iris, y_iris


def built_graphs(X: list[list[float]], y: list[int], my_per: Perceptron) -> None:
    """Plot the decision boundary (left) and errors per epoch (right)."""
    win, (left_gr, r_gr) = plt.subplots(1, 2, figsize=(12, 5))
    left_gr.set_title("Граница решения")
    point_x = [p[0] for p in X]
    point_y = [p[1] for p in X]
    left_gr.scatter(point_x, point_y, c=y, cmap="bwr")
    w1 = my_per.wight[0]
    w2 = my_per.wight[1]
    if w2 != 0:
        k = -w1 / w2
        b = -my_per.bias / w2
        x_min, x_max = left_gr.get_xlim()
        y_for_xmin = k * x_min + b
        y_for_xmax = k * x_max + b
        left_gr.plot([x_min, x_max], [y_for_xmin, y_for_xmax], color="green", linewidth=2)
        left_gr.set_ylim(min(point_y) - 1, max(point_y) + 1)
    # создание второго графика
    epochs_range = list(range(1, len(my_per.list_error) + 1))
    r_gr.plot(epochs_range, my_per.list_error, marker="o", color="red")
    r_gr.set_title("Падение ошибок при обучении")
    r_gr.set_xlabel("Номер эпохи")
    r_gr.set_ylabel("Количество ошибок")

    # Показываем окно
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Parse CLI arguments, train the perceptron and print the accuracy."""
    # 1. Считываем настройки из терминала
    parser = argparse.ArgumentParser(description="перцептрон")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples должно быть не меньше 2")
    if args.epochs < 1:
        parser.error("--epochs должно быть не меньше 1")

    print(f"Создаем {args.samples} точек")
    X_data, y_data = generate_points(args.samples)

    print(f"Учим перцептрон. шаг обучения: {args.lr} эпохи: {args.epochs}")
    my_per = Perceptron(n_features=2, learning_rate=args.lr, epochs=args.epochs)
    my_per.train(X_data, y_data)
    acc = my_per.score(X_data, y_data)
    print(f"Точность работы: {acc}")
    built_graphs(X_data, y_data, my_per)


if __name__ == "__main__":
    main()
