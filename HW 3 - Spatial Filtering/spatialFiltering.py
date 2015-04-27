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

def averagingFilter(image):
  """Each pixel becomes the average of its immediately surrounding pixels.
  We are doing a simple 3x3 box blur.
  
  Referencing below zero wraps around, so top and left sides will be blurred.
   
  We are not bothering with the right and bottom edges, because referencing
  above the image size results in a boundary error.
  """
  width, height = image.shape

  filteredImage = copy.deepcopy(image)

  # Avoid right, bottom edges.
  for i in range(width - 1):
    for j in range(height - 1):
       
      total = 0.0
      
      for i1 in range(i - 1, i + 2):
        for j1 in range(j - 1, j + 2):
          total = total + float(image[i1][j1])
      
      filteredImage[i][j] = float(total) / float(9)
  
  return filteredImage

def gaussianFilter(image):
  """Each pixel becomes the Gaussian-weighted average of nearby pixels.
  
  Referencing below zero wraps around, so top and left sides will be blurred.
   
  We are not bothering with the right and bottom edges, because referencing
  above the image size results in a boundary error.
  """
  width, height = image.shape

  filteredImage = copy.deepcopy(image)

  # Avoid right, bottom edges.
  for i in range(width - 2):
    for j in range(height - 2):
       
      # Mask from homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
      total = 0.0
      total += 1 * float(image[i-2][j+2])
      total += 4 * float(image[i-1][j+2])
      total += 7 * float(image[i-0][j+2])
      total += 4 * float(image[i+1][j+2])
      total += 1 * float(image[i+2][j+2])
      
      total += 4 * float(image[i-2][j+1])
      total += 16 * float(image[i-1][j+1])
      total += 26 * float(image[i-0][j+1])
      total += 16 * float(image[i+1][j+1])
      total += 4 * float(image[i+2][j+1])
      
      total += 7 * float(image[i-2][j+0])
      total += 26 * float(image[i-1][j+0])
      total += 41 * float(image[i-0][j+0])
      total += 26 * float(image[i+1][j+0])
      total += 7 * float(image[i+2][j+0])
      
      total += 4 * float(image[i-2][j-1])
      total += 16 * float(image[i-1][j-1])
      total += 26 * float(image[i-0][j-1])
      total += 16 * float(image[i+1][j-1])
      total += 4 * float(image[i+2][j-1])
      
      total += 1 * float(image[i-2][j-2])
      total += 4 * float(image[i-1][j-2])
      total += 7 * float(image[i-0][j-2])
      total += 4 * float(image[i+1][j-2])
      total += 1 * float(image[i+2][j-2])
      
      filteredImage[i][j] = total / float(273)
  
  return filteredImage
  
def medianFilter(image):
  """Each pixel becomes the median of its immediately surrounding pixels.
  We are doing a simple 5x5 median blur.
  
  Referencing below zero wraps around, so top and left sides will be blurred.
   
  We are not bothering with the right and bottom edges, because referencing
  above the image size results in a boundary error.
  """
  width, height = image.shape

  filteredImage = copy.deepcopy(image)

  # Avoid right, bottom edges.
  for i in range(width - 2):
    for j in range(height - 2):
       
      neighborhood = list()
      
      for i1 in range(i - 2, i + 3):
        for j1 in range(j - 2, j + 3):
          neighborhood.append(image[i1][j1])
      
      filteredImage[i][j] = numpy.median(neighborhood)
  
  return filteredImage
  
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
      
      filteredImage[i][j] = originalImage[i][j] + (1.5 * (total / 4.0))
  
  return filteredImage

def sobelXFilter(image):
  """Performs a horizontal Sobel operation.
  
  Referencing below zero wraps around, so top and left sides will be sharpened.
   
  We are not bothering with the right and bottom edges, because referencing
  above the image size results in a boundary error.
  """
  width, height = image.shape

  filteredImage = copy.deepcopy(image)
  #originalImage = copy.deepcopy(image)
  
  # Remove some noise before we begin.
  image = medianFilter(image)  
  
  # Avoid right, bottom edges.
  for i in range(width - 1):
    for j in range(height - 1):
       
      # Mask from en.wikipedia.org/wiki/Sobel_operator
      total = 0.0
      total += -1 * float(image[i - 1][j - 1])
      total +=  1 * float(image[i + 1][j - 1])
      total += -2 * float(image[i - 1][j])
      total +=  2 * float(image[i + 1][j])
      total += -1 * float(image[i - 1][j + 1])
      total +=  1 * float(image[i + 1][j + 1])
      
      #filteredImage[i][j] = originalImage[i][j] + (total / 6.0)
      filteredImage[i][j] = total / 6.0
  
  return filteredImage

def sobelYFilter(image):
  """Performs a vertical Sobel operation.
  
  Referencing below zero wraps around, so top and left sides will be sharpened.
   
  We are not bothering with the right and bottom edges, because referencing
  above the image size results in a boundary error.
  """
  width, height = image.shape

  filteredImage = copy.deepcopy(image)
  #originalImage = copy.deepcopy(image)

  # Remove some noise before we begin.  
  image = medianFilter(image)

  # Avoid right, bottom edges.
  for i in range(width - 1):
    for j in range(height - 1):
       
      # Mask from en.wikipedia.org/wiki/Sobel_operator
      total = 0.0
      total += -1 * float(image[i - 1][j - 1])
      total += -2 * float(image[i + 0][j - 1])
      total += -1 * float(image[i + 1][j - 1])
      total +=  1 * float(image[i - 1][j + 1])
      total +=  2 * float(image[i - 0][j + 1])
      total +=  1 * float(image[i + 1][j + 1])
      
      #filteredImage[i][j] = originalImage[i][j] + (total / 6.0)
      filteredImage[i][j] = total / 6.0
  
  return filteredImage

def sobelXYFilter(image):
  """
  Combines the Sobel X and Y filters to find all edges.
  """
  
  width, height = image.shape
  
  xFiltered = sobelXFilter(copy.deepcopy(image))
  yFiltered = sobelYFilter(copy.deepcopy(image))
  
  for i in range(width):
    for j in range(height):
      x = xFiltered[i][j]
      y = yFiltered[i][j]
      
      image[i][j] = math.sqrt((x ** 2) + (y ** 2))
  
  return image

def saveImage(image, filename):
  """Saves the image in the output directory with the filename given.
  """
  cv2.imwrite(OUTPUT_DIRECTORY + filename + IMAGE_FILE_EXTENSION, image)

def openImage(fileName):
  """Opens the image in the input directory with the filename given.
  """
  return cv2.imread(INPUT_DIRECTORY + fileName + IMAGE_FILE_EXTENSION, 0)
  
# Input images
inputForBlurring = 'fabio'
inputForSharpening = 'bball'

# Import image.
imageForBlurring = openImage(inputForBlurring)
imageForSharpening = openImage(inputForSharpening)

## Run filters on image, save.
#print("Averaging Filter...")
#saveImage(averagingFilter(imageForBlurring), inputForBlurring + 'Averaging')
#
#print("Gaussian Filter...")
#saveImage(gaussianFilter(imageForBlurring), inputForBlurring + 'Gauss')
#
#print("Median Filter...")
#saveImage(medianFilter(imageForBlurring), inputForBlurring + 'Median')

print("Laplacian Filter...")
saveImage(laplacianFilter(imageForSharpening), inputForSharpening + 'Laplace')

print("Sobel X Filter...")
saveImage(sobelXFilter(imageForSharpening), inputForSharpening + 'XSobel')

print("Sobel Y Filter...")
saveImage(sobelYFilter(imageForSharpening), inputForSharpening + 'YSobel')

print("Sobel XY Filter...")
saveImage(sobelXYFilter(imageForSharpening), inputForSharpening + 'XYSobel')

print("Done.")
