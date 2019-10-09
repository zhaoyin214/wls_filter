#%%
from skimage.io import imread
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#%%
filepath = "./img/flower.png"
image_name = os.path.split(filepath)[-1]
image_name = os.path.splitext(image_name)[0]

image = imread(filepath)
image_fine = imread(os.path.join("./output", image_name + "_fine.png"))
image_medium = imread(os.path.join("./output", image_name + "_medium.png"))
image_coarse = imread(os.path.join("./output", image_name + "_coarse.png"))
image_combined = imread(os.path.join("./output", image_name + "_combined.png"))

fig = plt.figure(figsize=(16, 12), facecolor="white")

ax = fig.add_subplot(2, 2, 1)
ax.imshow(image)
ax.axis("off")
ax.set_title("原始图像", color="black")

ax = fig.add_subplot(2, 2, 2)
ax.imshow(image_fine)
ax.axis("off")
ax.set_title("小尺度", color="black")

ax = fig.add_subplot(2, 2, 3)
ax.imshow(image_medium)
ax.axis("off")
ax.set_title("中尺度", color="black")

ax = fig.add_subplot(2, 2, 4)
ax.imshow(image_coarse)
ax.axis("off")
ax.set_title("大尺度", color="black")

fig = plt.figure(figsize=(12, 16), facecolor="white")

ax = fig.add_subplot(2, 1, 1)
ax.imshow(image)
ax.axis("off")
ax.set_title("原始图像", color="black")

ax = fig.add_subplot(2, 1, 2)
ax.imshow(image_combined)
ax.axis("off")
ax.set_title("多尺度滤波", color="black")

plt.show()


#%%
