import numpy as np
import torch
import matplotlib.pyplot as plt
from util import linear_subspace

# input
n = 1000
m = 200
A = np.random.random(size=(m, n)).astype(dtype=np.float32)
b = np.random.random(size=(m, 1)).astype(dtype=np.float32)
# input end

def solve(A: np.ndarray, b: np.ndarray, p: float = 2, num_steps: int = 10) -> np.ndarray:
    x_, Q2 = linear_subspace(A, b)
    x__torch = torch.from_numpy(x_).requires_grad_(False)
    Q2_torch = torch.from_numpy(Q2).requires_grad_(False)
    z_torch = torch.rand(size=(n - m, 1)).requires_grad_(True)
    optimizer = torch.optim.LBFGS(params=[z_torch, ], lr=0.1)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        x_torch = x__torch.__add__(Q2_torch.__matmul__(z_torch))
        objective = torch.sum(torch.abs(x_torch).__pow__(p))
        objective.backward()
        return objective

    for i in range(num_steps):
        loss = optimizer.step(closure).detach().numpy()
        print(i, loss)

    x_torch = x__torch.__add__(Q2_torch.__matmul__(z_torch))
    return x_torch.detach().numpy()


x_norm2 = solve(A, b, 2, num_steps=100)
x_norm1 = solve(A, b, 1, num_steps=100)


def draw_hist(x: np.ndarray, title: str="norm"):
    hist, edge = np.histogram(x, bins=101, range=[-0.1, +0.1])
    center = np.array([0.5 * (edge[i] + edge[i+1]) for i in range(len(hist))])
    plt.title(title)
    plt.xlabel("values")
    plt.ylabel("occurrences")
    plt.bar(center, hist, width=(center[1] - center[0]))
    plt.show()

draw_hist(x_norm2, "norm2")
draw_hist(x_norm1, "norm1")






pass
