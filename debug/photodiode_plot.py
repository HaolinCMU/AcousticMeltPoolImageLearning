# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 19:35:49 2022

@author: hlinl
"""


import numpy as np
import matplotlib.pyplot as plt


photo = np.load('F:/data/raw/photodiode/Layer0213_P200_V0250_C001H001S0001.npy')[0,:] # Special case: Layer 345, 213
photo_thrsld = 0.05
begin = np.where(photo>=photo_thrsld)[0][0]
end = np.where(photo>=photo_thrsld)[0][-1]

print(len(photo))

print(begin)
print(end)

plt.figure()
plt.plot(photo)
plt.show()
plt.close()

plt.figure()
plt.plot(photo[begin:end])
plt.show()
plt.close()

plt.figure()
plt.plot(photo[begin-5000:end+5000])
plt.show()
plt.close()

print(begin-5000)
print(end+5000)