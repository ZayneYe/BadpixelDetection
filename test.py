import numpy as np

a = np.array([1,2,3])
b = np.array([3,2,1])

c = np.stack((a, b),axis=0)
c = np.max(c,axis=0)
print(c)