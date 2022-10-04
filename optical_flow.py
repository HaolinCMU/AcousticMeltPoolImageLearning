
#%%
import re
from image_processing import *
import cv2
import matplotlib.pyplot as plt


area_threshold = 160
image_threshold = 200 # image shreshold : 0.8-1.0; 0.8*255 = 204

# calculate one track index
def optical_flow(path,plot = False):
    path = path
    num = int(re.findall(r"(\d{4}).png",path)[0])
    root = re.findall(r"(.*)\d{4}.png",path)
    for i in range(num,num-50,-1):
        file_path = root[0] + f"{i}"+'.png'
        if os.path.exists(path = file_path):
            image_process = Frame(file_path = root[0] + f"{i}"+'.png')
            area = image_process._meltpool_area
            if area < area_threshold:
                break
        else:
            break
    begin = i+1  #find begin number
    for j in range(num,num+50):
        file_path = root[0] + f"{j}"+'.png'
        if os.path.exists(path = file_path):
            image_process = Frame(file_path = root[0] + f"{j}"+'.png')
            area = image_process._meltpool_area
            if area <area_threshold:
                break 
        else:
            break
    end = j-1   #find end number



    # optical flow
    u = 0
    v = 0
    aspect_ratio_sum = 0
    # area_sum = 0
    for i in range(begin,end):
        prvs = cv2.imread(root[0] + f"{i}"+'.png')
        prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)
        prvs = np.float32(prvs) 
        prvs[prvs<image_threshold] = 0 
        next = cv2.imread(root[0] + f"{i+1}"+'.png')
        next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
        next = np.float32(next)
        next[next<image_threshold] = 0
        flow = cv2.calcOpticalFlowFarneback(prvs,next, cv2.CV_32F,0.5, 1, 5, 3, 5, 1.5, 0)#0.5, 3, 15, 3, 5, 1.2, 0  second:0.8, 15, 5, 10, 5, 0, 10
        image_process = Frame(file_path = root[0] + f"{i}"+'.png')
        length = image_process.meltpool_length
        width = image_process.meltpool_width
        aspect_ratio = length/width if width != 0. else 0.
        aspect_ratio_sum += np.exp(aspect_ratio)
        u += np.mean(flow[:,:,0])*np.exp(aspect_ratio)
        v += np.mean(flow[:,:,1])*np.exp(aspect_ratio)
    u /= aspect_ratio_sum   # exp(aspect_ratio) weighted
    v /= aspect_ratio_sum
    vec = np.array((u,v))
    normalized = vec / np.sqrt(np.sum(vec**2))
    # print(normalized)



    # plot
    if plot == True:
        image_process = Frame(file_path = path,intensity_threshold=(0.8, 1.))
        center_point = image_process.meltpool_center_pt
        prvs_1 = cv2.imread(path)
        prvs_1 = cv2.cvtColor(prvs_1, cv2.COLOR_BGR2GRAY)
        prvs_1 = np.float32(prvs_1)
        plt.imshow(prvs_1,cmap = "gray")
        plt.arrow(center_point[1], center_point[0], normalized[0]*50, normalized[1]*50,
                  head_width=20, head_length=20,color = "red") 


    return normalized[0],normalized[1]

# %%
if __name__ == "__main__":
    u, v = optical_flow('/Users/puxueting/Desktop/raw_image_data/Layer003_\
                        Section_01_S0001/Layer003_Section_01_S0001003463.png')