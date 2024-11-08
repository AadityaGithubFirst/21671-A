import numpy as np
A = np.array([[25, 30, 45, 60], [43, 44, 12, 32]])
B = np.array([[12, 21, 32], [42,34, 53], [78,19, 90], [93,37,89]])
def multiply(A:np.array, B:np.array):
    if np.shape(A)[1] != np.shape(B)[0]:
        return "The matrices are not able to be multiplied"
    C = np.zeros([np.shape(A)[0], np.shape(B)[1]])
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(B)[1]):
            for k in range(np.shape(A)[1]):
                C[i][j] += A[i][k]*B[k][j]
    return C
print(multiply(A,B))
print(multiply(B, A))


