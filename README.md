# sparsest-solution-underdetermined-linear-system

optimize norm with underdetermined system equality constraint

## problem statement

```
Minimize ||x||_p
Given Ax=b
where   x \in R^n
        A \in R^{m \times n}
        b \in R^m
        p \in R_+
```

## algorithm

### unconstrained optimization (L_p norm, p >= 1)

```
Minimize ||Ax-b||_2^2 + ||x||_p^p
```

```
Let z \in R^{n-m} be an arbitrary vector.
Represent the solution of Ax=b by x = A* z + b* // see boyd convex optimization
The problem becomes minimizing ||A*z + b*||_p
```

### linear programming (L_1 norm)

```
# idea
Let y \in R^{n} with 2 additional constraints
y \geq x and y \geq -x (element-wise)
Let u = [x, y] \in R^{2n}, the feasible set is a polyhedron.
Minimize sum of y, get x

# explanation
y \geq x and y \geq -x constraint y \geq |x|
Let u1 = [x1, y1] be the minimizer.
It is easy to prove that minimal y, y1 = |x1|
Hence, the LP formulation yeilds the same solution as the original problem.
```

## results (m = 400, n = 2000, random A b)

### L2 norm sparsity

![norm2](https://raw.githubusercontent.com/khanhhhh/sparse-uls/main/assets/norm2.png)

### L1.001 norm sparsity

![norm1001](https://raw.githubusercontent.com/khanhhhh/sparse-uls/main/assets/norm1001.png)

### L1 norm sparsity

![norm1](https://raw.githubusercontent.com/khanhhhh/sparse-uls/main/assets/norm1.png)

## Packaging

```bash
rm -rf dist/*
python setup.py sdist bdist_wheel
twine upload dist/*

```

## Useful links

- https://pypi.org/project/sparse-uls/

- https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf (page 682)

- https://packaging.python.org/tutorials/packaging-projects/

- https://dzone.com/articles/executable-package-pip-install
