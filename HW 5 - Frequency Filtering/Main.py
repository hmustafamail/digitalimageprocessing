# Mustafa Hussain
# Digital Image Processing with Dr. Anas Salah Eddin
# FL Poly, Spring 2015
#
# Homework 4: Discrete Fourier Transform
#
# USAGE NOTES:
#
# Written in Python 2.7
#
# Please ensure that the script is running as the same directory as the images 
# directory!

import cv2
import copy
import numpy
import math

INPUT_DIRECTORY = 'input/'
OUTPUT_DIRECTORY = 'output/'
IMAGE_FILE_EXTENSION = '.JPG'

MAX_INTENSITY = 255 # 8-bit images



#==============================================================================
# Filtering functions
#==============================================================================

def filter(image, mask):
  """
  Applies frequency mask to image.
  """
  
  fourierImage = numpy.fft.fftshift(numpy.fft.fft2(image))
  
  filteredFourier = fourierImage * mask
  
  return reconstruct(filteredFourier, normalize=True)
  

def laplace(image):
  """
  TODO: Laplace filter.
  """
  
  width, height = image.shape
  
  fourierImage = numpy.fft.fftshift(numpy.fft.fft2(image))  
  
  # Because mathemeticians insist on being cryptic.
  M, N = width, height
  
  laplaceVersion = copy.deepcopy(fourierImage)
  
  for i in range(width):
    for j in range(height):
      u = i
      v = j
      laplaceVersion[i][j] = (1 - ((u - (M / 2)) ** 2 + (v - (N / 2)) ** 2)) * fourierImage[u][v]
  
  return reconstruct(laplaceVersion, normalize=True)
  
  
  
  
#==============================================================================
# Image reconstruction functions
#==============================================================================

def reconstruct(fourierImage, normalize=True):
  """Reconstructs from the Fourier domain into something you can see.
  """
  reconstructed = numpy.abs(numpy.fft.ifft2(fourierImage)) 
  
  if normalize:
    reconstructed = reconstructed / numpy.max(reconstructed)
    reconstructed = reconstructed * MAX_INTENSITY
  
  return reconstructed


#==============================================================================
# Image import/export
#==============================================================================

def saveImage(image, filename, filetype = IMAGE_FILE_EXTENSION):
  """Saves the image in the output directory with the filename given.
  """
  cv2.imwrite(OUTPUT_DIRECTORY + filename + filetype, image)

def openImage(fileName, filetype = IMAGE_FILE_EXTENSION):
  """Opens the image in the input directory with the filename given.
  """
  return cv2.imread(INPUT_DIRECTORY + fileName + filetype, 0)


# Import images.
image1 = openImage('testImage1')
idealLowpassMask4 = openImage('idealLowpassMask4', '.png')
idealLowpassMask32 = openImage('idealLowpassMask32', '.png')
idealLowpassMask64 = openImage('idealLowpassMask64', '.png')
idealHighpassMask4 = openImage('idealHighpassMask4', '.png')
idealHighpassMask32 = openImage('idealHighpassMask32', '.png')
idealHighpassMask64 = openImage('idealHighpassMask64', '.png')
gaussianLowpassMask4 = openImage('gaussianLowpassMask4', '.png')
gaussianLowpassMask32 = openImage('gaussianLowpassMask32', '.png')
gaussianLowpassMask64 = openImage('gaussianLowpassMask64', '.png')
gaussianHighpassMask4 = openImage('gaussianHighpassMask4', '.png')
gaussianHighpassMask32 = openImage('gaussianHighpassMask32', '.png')
gaussianHighpassMask64 = openImage('gaussianHighpassMask64', '.png')

# Process images, save.
# Ideal Lowpass
print("Performing Ideal Lowpass Filtering...")

filtered1 = filter(image1, idealLowpassMask4)
filtered2 = filter(image1, idealLowpassMask32)
filtered3 = filter(image1, idealLowpassMask64)

saveImage(filtered1, 'idealLowpass1')
saveImage(filtered2, 'idealLowpass2')
saveImage(filtered3, 'idealLowpass3')


# Ideal Highpass
print("Performing Ideal Highpass Filtering...")

filtered1 = filter(image1, idealHighpassMask4)
filtered2 = filter(image1, idealHighpassMask32)
filtered3 = filter(image1, idealHighpassMask64)

saveImage(filtered1, 'idealHighpass1')
saveImage(filtered2, 'idealHighpass2')
saveImage(filtered3, 'idealHighpass3')


# Gaussian Lowpass
print("Performing Gaussian Lowpass Filtering...")

filtered1 = filter(image1, gaussianLowpassMask4)
filtered2 = filter(image1, gaussianLowpassMask32)
filtered3 = filter(image1, gaussianLowpassMask64)

saveImage(filtered1, 'gaussianLowpass1')
saveImage(filtered2, 'gaussianLowpass2')
saveImage(filtered3, 'gaussianLowpass3')


# Gaussian Highpass
print("Performing Gaussian Highpass Filtering...")

filtered1 = filter(image1, gaussianHighpassMask4)
filtered2 = filter(image1, gaussianHighpassMask32)
filtered3 = filter(image1, gaussianHighpassMask64)

saveImage(filtered1, 'gaussianHighpass1')
saveImage(filtered2, 'gaussianHighpass2')
saveImage(filtered3, 'gaussianHighpass3')


# Laplace 
print("Performing Laplace Filtering...")

filtered1 = laplace(image1)

saveImage(filtered1, 'laplace')

# Laplace of Gaussian
print("Performing Laplace of Gaussian Filtering...")

filtered1 = laplace(filter(image1, gaussianLowpassMask4))
filtered2 = laplace(filter(image1, gaussianLowpassMask32))
filtered3 = laplace(filter(image1, gaussianLowpassMask64))

saveImage(filtered1, 'laplaceOfGaussian1')
saveImage(filtered2, 'laplaceOfGaussian2')
saveImage(filtered3, 'laplaceOfGaussian3')


print("Done.")
