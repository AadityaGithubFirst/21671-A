import numpy as np
from numpy.linalg import solve, norm, cond
from scipy import linalg

def hilbert_analysis(n_max):
    relative_error=0
    n=2
    while(relative_error<100):
        H = linalg.hilbert(n)        
        x = np.ones(n)
        b = H @ x
        x_hat = solve(H, b)
        r = b - H @ x_hat
        delta_x = x - x_hat
        r_norm = norm(r, np.inf)
        error_norm = norm(delta_x, np.inf)
        relative_error = error_norm / norm(x, np.inf)
        cond_num = cond(H, np.inf)
        print(f"n = {n}")
        print(f"Residual norm: {r_norm:.2e}")
        print(f"Error norm: {error_norm:.2e}")
        print(f"Relative error: {relative_error}")
        print(f"Condition number: {cond_num:.2e}")
        print()
        
        if relative_error >= 1:
            print(f"Relative error reached 100% at n = {n}")
            break
        n+=1
hilbert_analysis(20)
