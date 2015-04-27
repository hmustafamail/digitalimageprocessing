# Mustafa Hussain
# Digital Image Processing with Dr. Anas Salah Eddin
# FL Poly, Spring 2015
#
# Homework 1: Invert an Image

# USAGE NOTES:
# Please ensure that the script is running as the same directory as robin.jpg!

from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import numpy
import copy

# Import image.
image = Image.open('robin.jpg')

colors = 256 # 8-bit image
width, height = image.size

# Convert image to NumPy array.
image = numpy.array(image)

# ...and keep a copy.
oldImage = copy.deepcopy(image)

# Invert image.
for i in range(width):
    for j in range(height):
        image[i][j] = colors - image[i][j]

# Show the before-and-after.
plt.figure()
io.imshow(oldImage)

plt.figure()
io.imshow(image)

io.show()
