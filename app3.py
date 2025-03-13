from skimage.metrics import structural_similarity
import cv2
import numpy as np
import PIL
import PIL.Image as img

img1 = img.open('imag1.png')
img2 = img.open('imag2.png')

# Determine the common dimensions (smallest width and height)
common_width = min(img1.width, img2.width)
common_height = min(img1.height, img2.height)

# Resize images
img1_resized = img1.resize((300, 400), PIL.Image.ANTIALIAS)
img2_resized = img2.resize((300, 400), PIL.Image.ANTIALIAS)

# Save resized images
img1_resized.save('image1.png')
img2_resized.save('image2.png')

# Load images
image1 = cv2.imread('image1.png')
image2 = cv2.imread('image2.png')

difference = cv2.subtract(image1, image2)

# color the mask red
Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
difference[mask != 255] = [0, 0, 255]

# add the red mask to the images to make the differences obvious
image1[mask != 255] = [0, 0, 255]
image2[mask != 255] = [0, 0, 255]

# store images
cv2.imwrite('diffOverImage1.png', image1)
cv2.imwrite('diffOverImage2.png', image1)
cv2.imwrite('diff.png', difference)