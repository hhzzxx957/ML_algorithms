import numpy as np
import pandas as pd
# define a matrix
# A = np.array([[1, 2], [3, 4], [5, 6]])

columns = ["var","skewness","curtosis","entropy","class"]
A = pd.read_csv("./data/data_banknote_authentication.txt",index_col=False, names = columns)
print(A.head())
# calculate the mean of each column
M = np.mean(A.T, axis=1)
# center columns by subtracting column means
C = A - M
# calculate covariance matrix of centered matrix
V = np.cov(C.T)
# eigendecomposition of covariance matrix
e_vals, e_vecs = np.linalg.eig(V)
# print(e_vals)
# print(e_vecs)

# project data
P = e_vecs.T.dot(C.T)
print(P.T)

# explained variance
var_exp = [(i / sum(e_vals))*100 for i in sorted(e_vals, reverse=True)]
print(np.cumsum(var_exp))

from matplotlib import pyplot as plt
plt.plot(P[0], P[1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
# plt.plot(P[0,20:40], P[1,20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlim([-20,20])
plt.ylim([-20,20])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples with class labels')

plt.show()
