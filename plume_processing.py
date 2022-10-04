
# %%
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import colorsys
from PIL import Image, ImageEnhance
# Function to extract frames
def FrameCapture(path):
      
    # Path to video file
    vidObj = cv2.VideoCapture(path)
  
    # Used as counter variable
    count = 0
  
    # checks whether frames were extracted
    success = 1
  
    while success:
  
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
  
        # Saves the frames with frame-count
        cv2.imwrite("lof_frame%d.jpg" % count, image)
  
        count += 1
  
  
# Calling the function
# tranform video to frames
FrameCapture("/Users/puxueting/Desktop/with_powder/L138_L141_P0100_V1200_A45_LOF.MOV")

# get difference
img1 = cv2.imread("/Users/puxueting/Desktop/AcousticMeltPoolImageLearning-main/lof_frame6733.jpg").astype(np.float32)
img1 = cv2.normalize(img1, None, 0, 1, cv2.NORM_MINMAX)
img2 = cv2.imread("/Users/puxueting/Desktop/AcousticMeltPoolImageLearning-main/lof_frame6762.jpg").astype(np.float32)
img2 = cv2.normalize(img2, None, 0, 1, cv2.NORM_MINMAX)
dif = img2-img1
dif = np.absolute(dif)
# enhance image
im = Image.fromarray((dif*255).astype('uint8'), 'RGB')
enhancer = ImageEnhance.Brightness(im)
factor = 10 #gives original image
im_output = enhancer.enhance(factor)
plt.imshow(im_output)

