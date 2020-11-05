from typing import Iterator, Dict, Callable, Union

import numpy as np
import torch

from sparse_uls.util import linear_subspace, lbfgs_optimizer


def solve(A: np.ndarray, b: np.ndarray, p: float = 1.0,
          num_steps: int = 10,
          optimizer: Callable[
              [Union[Iterator[torch.Tensor], Iterator[Dict]]], torch.optim.Optimizer] = lbfgs_optimizer(),
          ) -> np.ndarray:
    m, n = A.shape
    x_, Q2 = linear_subspace(A, b)
    x__torch = torch.from_numpy(x_).requires_grad_(False)
    Q2_torch = torch.from_numpy(Q2).requires_grad_(False)
    z_torch = torch.rand(size=(n - m, 1)).requires_grad_(True)
    optim = optimizer([z_torch, ])

    def closure() -> torch.Tensor:
        optim.zero_grad()
        x_torch = x__torch.__add__(Q2_torch.__matmul__(z_torch))
        objective = torch.sum(torch.abs(x_torch).__pow__(p))
        objective.backward()
        return objective

    for i in range(num_steps):
        optim.step(closure)

    x_torch = x__torch.__add__(Q2_torch.__matmul__(z_torch))
    return x_torch.detach().numpy()
