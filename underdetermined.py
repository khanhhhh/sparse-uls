import numpy as np
import torch

from util import linear_subspace


def solve(A: np.ndarray, b: np.ndarray, p: float = 2, num_steps: int = 10) -> np.ndarray:
    m, n = A.shape
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
