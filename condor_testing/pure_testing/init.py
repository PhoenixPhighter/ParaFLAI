import numpy as np

size = 100
no_of_jobs = 10

for i in range(0, no_of_jobs):
   filenameA = "inputA%s.npy" % i
   filenameB = "inputB%s.npy" % i

   A = np.random.rand(size, size)
   B = np.random.rand(size, size)

   np.save(filenameA, A)
   np.save(filenameB, B)