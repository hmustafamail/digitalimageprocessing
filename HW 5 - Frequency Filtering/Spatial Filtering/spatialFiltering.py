# Mustafa Hussain
# Digital Image Processing with Dr. Anas Salah Eddin
# FL Poly, Spring 2015
#
# Homework 3: Spatial Filtering
#
# USAGE NOTES:
#
# Written in Python 2.7
#
# Please ensure that the script is running as the same directory as the images 
# directory!

import cv2
import copy
#import matplotlib.pyplot as plt
import numpy
import math
#from skimage import exposure

INPUT_DIRECTORY = 'input/'
OUTPUT_DIRECTORY = 'output/'
IMAGE_FILE_EXTENSION = '.JPG'

MAX_INTENSITY = 255 # 8-bit images

def laplacianFilter(image):
  """Approximates the second derivative, bringing out edges.
  
  Referencing below zero wraps around, so top and left sides will be sharpened.
   
  We are not bothering with the right and bottom edges, because referencing
  above the image size results in a boundary error.
  """
  width, height = image.shape

  filteredImage = copy.deepcopy(image)
  originalImage = copy.deepcopy(image)
  
  # Avoid right, bottom edges.
  for i in range(width - 1):
    for j in range(height - 1):
       
      # Mask from homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
      total = 0.0
      total += -1 * float(image[i][j + 1])
      total += -1 * float(image[i - 1][j])
      total += 4 * float(image[i][j])
      total += -1 * float(image[i + 1][j])
      total += -1 * float(image[i][j - 1])
      
      filteredImage[i][j] = total / 9.0
  
  filteredImage = (filteredImage / numpy.max(filteredImage)) * MAX_INTENSITY
  return filteredImage

def saveImage(image, filename):
  """Saves the image in the output directory with the filename given.
  """
  cv2.imwrite(OUTPUT_DIRECTORY + filename + IMAGE_FILE_EXTENSION, image)

def openImage(fileName):
  """Opens the image in the input directory with the filename given.
  """
  return cv2.imread(INPUT_DIRECTORY + fileName + IMAGE_FILE_EXTENSION, 0)
  
# Input images
inputForSharpening = 'testImage1'

# Import image.
imageForSharpening = openImage(inputForSharpening)


print("Laplacian Filter...")

filtered = laplacianFilter(imageForSharpening)

saveImage(filtered, inputForSharpening + 'Laplace')


print("Done.")
