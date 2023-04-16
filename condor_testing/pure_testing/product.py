import numpy as np

filenameA = "inputA.npy"
filenameB = "inputB.npy"
filenameC = "output.npy"

A = np.load(filenameA)
B = np.load(filenameB)

C = np.matmul(A, B)

np.save(filenameC, C)