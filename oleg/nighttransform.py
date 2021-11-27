import numpy as np
from imageio import imread, imwrite
import cv2
# The photo we want to convert.
img = imread('photo.jpg')

print(type(img))
arr = img*np.array([0.1, 0.2, 0.5])

arr2 = (255*arr/arr.max()).astype(np.uint8)

imwrite('night.png', arr2)

mg2 = cv2.imread('night.png')

gamma = 2    #More the gamma more will be darkness

gamma_img = np.array(255*(img2/255) **gamma, dtype = 'uint8')

cv2.imwrite('night_final.png', gamma_img)