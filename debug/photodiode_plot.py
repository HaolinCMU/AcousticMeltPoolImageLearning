# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 19:35:49 2022

@author: hlinl
"""


import numpy as np
import matplotlib.pyplot as plt


photo = np.load('F:/data/raw/photodiode/Layer0021_P200_V0250_C001H001S0001.npy')[0,:] # Special case: Layer 345, 213
photo_thrsld = 0.2
begin = np.where(photo>=photo_thrsld)[0][0]
end = np.where(photo>=photo_thrsld)[0][-1]

clip_length = 100
clip_stride = 50
clip_num = 7

start_ind_list = [int(i*clip_stride) for i in range(clip_num)] + [22000] # For Layer 0021: 20000, 22000, 46200. 

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


for ind in start_ind_list: 
    plt.figure()
    plt.plot(photo[begin+ind:begin+ind+clip_length])
    plt.show()
    plt.close()
    

# plt.figure()
# plt.plot(photo[begin:begin+38780])
# plt.show()
# plt.title("Clip - 5")
# plt.close()

print(begin-5000)
print(end+5000)