from numpy.random import rand
from random import seed
from time import time
from numpy.linalg import solve, norm
import matplotlib.pyplot as plt
def solve_and_measure(n):
    A = rand(n, n)
    x = rand(n, 1)
    b = A @ x
    start_time = time()
    x_hat = solve(A, b)
    solve_time = time() - start_time
    x_norm = norm(x, 2)
    relative_error = norm(x - x_hat, 2) / x_norm
    residual_norm = norm(b - A @ x_hat, 2) / norm(b, 2)
    return [solve_time, x_norm, relative_error, residual_norm]
n = [100, 200, 500, 800, 1000, 2000, 5000, 8000, 10000, 15000, 20000]
results = {"n":[], "solve_time":[], "x_norm":[], "relative_error":[], "residual_norm":[]}
for i in n:
    a = solve_and_measure(i)
    results["n"].append(i)
    results["solve_time"].append(a[0])
    results["x_norm"].append(a[1])
    results["relative_error"].append(a[2])
    results["residual_norm"].append(a[3])
    print(results)
fig, ax = plt.subplots()
a = results["solve_time"]
ax.plot(n, a)
plt.show()