#%%
import numpy as np
from collections import defaultdict
# %%
def data_parse(path):
    sensor_1 = defaultdict(dict)
    sensor_2 = defaultdict(dict)
    sensor_3 = defaultdict(dict)

    for data_num in range(3):
        para_index = 0
        for n in range(21,294,4):
            layer_num = data_num*276+n
            if len(str(layer_num)) == 2:
            # if layer_num > 100:
            #     break
                data = np.load(path + f'/Acoustics_Layer000{layer_num}.npy')
            else:
                data = np.load(path + f'/Acoustics_Layer00{layer_num}.npy')
            sensor_1[f'para_{para_index}'][data_num] = list(data[0,:])
            sensor_2[f'para_{para_index}'][data_num] = list(data[1,:])
            sensor_3[f'para_{para_index}'][data_num] = list(data[2,:])
            para_index += 1
    # print('finished')
    return (sensor_1,sensor_2,sensor_3)
# %%
if __name__ == "__main__":
    sensor_1,sensor_2,sensor_3 = data_parse('/Users/puxueting/Desktop/to_Xueting')
# %%
