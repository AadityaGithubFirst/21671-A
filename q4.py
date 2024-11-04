import numpy as np
from time import time
from numpy import eye, zeros_like
from numpy.random import rand
import matplotlib.pyplot as plt
def lu_factorization(A):
    n = A.shape[0]
    L = eye(n)
    U = A.copy()
    for k in range(n-1):
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    return L, U
def forward_substitution(L, b):
    n = L.shape[0]
    y = zeros_like(b, dtype=float)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y
def backward_substitution(U, y):
    n = U.shape[0]
    x = zeros_like(y, dtype=float)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x
def solve_linear_system(A, b):
    L, U = lu_factorization(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x
def time_lu_solve(n):
    A = rand(n, n)
    b = rand(n)
    start_time = time()
    L, U = lu_factorization(A)
    lu_time = time() - start_time
    start_time = time()
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    solve_time = time() - start_time
    return lu_time, solve_time
A_test = np.array([[1, 1, 0], [2, 1, -1], [3, -1, -1]], dtype=float)
b_test = np.array([2, 1, 1], dtype=float)
x_test = solve_linear_system(A_test, b_test)
print("Test solution:", x_test)
print("Verification:", np.allclose(A_test @ x_test, b_test))
n_values = np.logspace(2, 4.39794000897, 10, dtype=int)
lu_times = []
solve_times = []
for n in n_values:
    lu_time, solve_time = time_lu_solve(n)
    lu_times.append(lu_time)
    solve_times.append(solve_time)
    print(f"n = {n}: LU factorization time = {lu_time:.6f}s, Solve time = {solve_time:.6f}s")
plt.figure(figsize=(10, 6))
plt.loglog(n_values, lu_times, 'bo-', label='LU Factorization')
plt.loglog(n_values, solve_times, 'ro-', label='Triangular Solve')
plt.xlabel('Matrix Size (n)')
plt.ylabel('Time (seconds)')
plt.title('LU Factorization and Triangular Solve Times')
plt.legend()
plt.grid(True)
plt.savefig('lu_solve_times.png')
plt.show()