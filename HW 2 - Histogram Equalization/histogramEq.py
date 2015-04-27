# Mustafa Hussain
# Digital Image Processing with Dr. Anas Salah Eddin
# FL Poly, Spring 2015
#
# Homework 2: Histogram Equalization
#
# USAGE NOTES:
#
# Written in Python 2.7
#
# Please ensure that the script is running as the same directory as the images 
# directory!

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

def histogramEqualize(image, maxIntensity):
  width, height = image.shape

  # Get cdf from image.
  cdf, binCenters = exposure.cumulative_distribution(image, maxIntensity)
  binCenters = binCenters.tolist()
  
  # Intensity transformation: Each pixel becomes the cumulative probability
  # that its intensity will show up, multiplied by the intended maximum 
  # intensity.
  for i in range(width):
    for j in range(height):
        
        try:
          probability = cdf[binCenters.index(image[i][j])]
        except:
          probability = 1
        
        image[i][j] = int(probability * maxIntensity)
  
  return image, binCenters, cdf

# Input images
inputImageNames = ['fabioDark', 'fabioBright', 'fabioLowContrast', 'fabioHighContrast']
maxIntensity = 255 # 8-bit images

inputDirectory = 'input/'
outputDirectory = 'output/'
imageFileExtension = '.JPG'

for imageName in inputImageNames:
  # Import image.
  image = cv2.imread(inputDirectory + imageName + imageFileExtension, 0)

  print "Processing", imageName, "(", inputImageNames.index(imageName) + 1, "out of", len(inputImageNames), ")..."
  
  # Record histogram of original image
  fabioHist = plt.hist(image)
  plt.axis([0, 255, 0, 350])
  plt.savefig(outputDirectory + imageName + 'Histogram' + imageFileExtension, bbox_inches='tight')
  plt.clf()

  # Run histogram equalization on image.
  image, binCenters, cdf = histogramEqualize(image, maxIntensity)
  
  # Save histogram equalized image.
  cv2.imwrite(outputDirectory + imageName + 'HistogramEqualized' + imageFileExtension, image)

  # Record histogram of new image
  fabioHist = plt.hist(image)
  plt.axis([0, 255, 0, 350])
  plt.savefig(outputDirectory + imageName + 'HistogramEqualizedHistogram' + imageFileExtension, bbox_inches='tight')
  plt.clf()
  
  # Save plot of transformation function.
  transFn = plt.scatter(binCenters, cdf)
  plt.axis([0, 255, 0, 1])
  plt.savefig(outputDirectory + imageName + 'TransformationFunction' + imageFileExtension, bbox_inches='tight')
  plt.clf()

print("Done.")
