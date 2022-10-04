#%%
from pyts.approximation import PiecewiseAggregateApproximation 
from pyts.preprocessing import MinMaxScaler 
import numpy as np 
import matplotlib.pyplot as plt
import numpy as np


#%%
def smallClips(x):

    transformer = PiecewiseAggregateApproximation(window_size=2) 
    result = transformer.transform(x) 

    # Scaling in interval [0,1] 
    scaler = MinMaxScaler() 
    scaled_X = scaler.transform(result) 

    # Transfer to polar coordinate
    arccos_X = np.arccos(scaled_X[1]) 


    field = [a+b for a in arccos_X for b in arccos_X] 
    gram = np.cos(field).reshape(-1,scaled_X.shape[1]) 
    return gram

#plot
# fig,ax = plt.subplots(3,3,figsize = (6,6))
# for i in range(3):
#     for j in range(3):
#         ax[i,j].imshow(smallClips(3*i+j))

if __name__ == "__main__":
    data = np.load('/Users/puxueting/Downloads/Layer0021_P200_V0250_C001H001S0001 (1).npy')
    # x is index, y is time series value
    x = np.arange(0,data[0].shape[0])
    y = data[0]

    n = len(x)
    new_list = []
    for i in range((n//200) -1):
        new_list.append([list(x[i*200:i*200+200]),list(y[i*200:i*200+200])])
    gram = smallClips(new_list[0])
# %%
