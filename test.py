import numpy as np
import matplotlib.pyplot as plt
from sparse_uls.uls import solve

from sparse_uls.util import lbfgs_optimizer
n = 1000
m = 200
A = np.random.random(size=(m, n)).astype(dtype=np.float32)
b = np.random.random(size=(m, 1)).astype(dtype=np.float32)


x_norm2 = solve(A, b, 2, num_steps=100, optimizer=lbfgs_optimizer(lr=0.1))
x_norm1 = solve(A, b, 1, num_steps=100, optimizer=lbfgs_optimizer(lr=0.1))


def draw_hist(x: np.ndarray, title: str = "norm"):
    hist, edge = np.histogram(x, bins=101, range=[-0.1, +0.1])
    center = np.array([0.5 * (edge[i] + edge[i + 1]) for i in range(len(hist))])
    plt.title(title)
    plt.xlabel("values")
    plt.ylabel("occurrences")
    plt.bar(center, hist, width=(center[1] - center[0]))
    plt.show()


draw_hist(x_norm2, "norm2")
draw_hist(x_norm1, "norm1")

pass
