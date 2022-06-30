
# %%
from image_processing import *
frame = Frame('C:/Users/hlinl/Desktop/acoustic_image_learning/data/raw_image_data/Layer003/Layer003_703.png')

plt.figure()
plt.imshow(frame.frame, cmap='gray')

plt.figure()
plt.imshow(frame.plume_image, cmap='gray')
# %%