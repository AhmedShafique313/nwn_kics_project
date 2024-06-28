from scipy.sparse import csr_matrix
import numpy as np

arr = np.array([0,0,0,0,0,1,1,0,2])
print(csr_matrix(arr))

arr1 = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
print(csr_matrix(arr1).data)


arr2 = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
print(csr_matrix(arr2).count_nonzero())

arr3 = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
mat = csr_matrix(arr3)
mat.eliminate_zeros()
print(mat)