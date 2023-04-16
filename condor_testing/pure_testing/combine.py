import numpy as np

size = 1000
no_of_jobs = 100
sum = np.zeros(size)

for i in range(0, no_of_jobs):
   filename = "output%s.npy" % i
   C = np.load(filename)
   sum = sum + C

print(np.linalg.norm(sum))