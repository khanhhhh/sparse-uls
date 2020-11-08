import time

import numpy as np
import matplotlib.pyplot as plt
from sparse_uls.uls import solve

np.random.seed(1234)

def draw_hist(x: np.ndarray, title: str = "norm"):
    hist, edge = np.histogram(x, bins=101, range=[-0.1, +0.1])
    center = np.array([0.5 * (edge[i] + edge[i + 1]) for i in range(len(hist))])
    plt.title(title)
    plt.xlabel("values")
    plt.ylabel("occurrences")
    plt.bar(center, hist, width=(center[1] - center[0]))
    plt.show()

def norm_p(x: np.ndarray, p: float=2.0) -> float:
    return np.sum(np.abs(x)**p)**(1/p)

n = 1000
m = 200
A = np.random.random(size=(m, n)).astype(dtype=np.float64)
b = np.random.random(size=(m)).astype(dtype=np.float64)

dp = 1
div = 2
while True:
    p = 1 + dp
    t0 = time.time()
    x = solve(A, b, p)
    t1 = time.time()
    print(f"L^{p} time: {t1-t0}")
    print(f"\t{np.max(np.abs(A @ x - b))}")
    print(f"\t{norm_p(x, p) / x.shape[0]}")
    draw_hist(x, f"L^{p} norm")
    dp /= div

pass
