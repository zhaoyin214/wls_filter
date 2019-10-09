#%%
from skimage.io import imread
import matplotlib.pyplot as plt
import os


#%%
filepath = "./img/flower.png"
image_name = os.path.split(filepath)[-1]
image_name = os.path.splitext(image_name)[0]

image = imread(filepath)
image_fine = imread(os.path.join("./output", image_name + "_fine.png"))
image_medium = imread(os.path.join("./output", image_name + "_medium.png"))
image_coarse = imread(os.path.join("./output", image_name + "_coarse.png"))

fig = plt.figure(figsize=(16, 12))

ax = fig.add_subplot(2, 2, 1)
ax.imshow(image)
ax.axis("off")

ax = fig.add_subplot(2, 2, 2)
ax.imshow(image_fine)
ax.axis("off")

ax = fig.add_subplot(2, 2, 3)
ax.imshow(image_medium)
ax.axis("off")

ax = fig.add_subplot(2, 2, 4)
ax.imshow(image_coarse)
ax.axis("off")

plt.show()



#%%
