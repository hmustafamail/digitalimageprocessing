# Mustafa Hussain
# Digital Image Processing with Dr. Anas Salah Eddin
# FL Poly, Spring 2015
#
# Homework 4: Discrete Fourier Transform (TIMING ONLY)
#
# USAGE NOTES:
#
# Written in Python 2.7
#
# Please ensure that the script is running as the same directory as the images 
# directory!

import numpy
import math
import time
import csv

INPUT_DIRECTORY = 'input/'
OUTPUT_DIRECTORY = 'output/'
IMAGE_FILE_EXTENSION = '.JPG'

MAX_INTENSITY = 255 # 8-bit images

NUM_SAMPLES = 10

#==============================================================================
# Discrete Fourier Transform functions
#==============================================================================

def singleRowDFT(row):
  """
  Computes the Discrete Fourier Transform of a 1-D row of numbers.
  """
  
  # Thanks nayuki.io/how-to-implement-the-discrete-fourier-transform
  #X = numpy.zeros(len(row))  
  X = []  
  
  n = float(len(row))
  
  for k in range(len(row)):
    currentSum = 0
    t = 0  
    
    while t < n:
      currentOperand = row[t] * numpy.exp((-2 * math.pi * 1j * k) / n)
      currentSum = currentSum + currentOperand
      t = t + 1
      
    #X[k] = currentSum
    X.append(currentSum)
      
  return X

def myDFT(image):
  """
  Computes the Discrete Fourier Transform of an image.
  """
    
  horizontalDFT = numpy.zeros(image.shape)  
  verticalAndHorizontalDFT = numpy.zeros(image.shape)
  
  # First, do the rows.
  counter = 0
  for row in image:
    horizontalDFT[counter] = singleRowDFT(row)
    counter = counter + 1
    
  # Next, do the columns.
  counter = 0
  for column in image.T:    
    verticalAndHorizontalDFT[counter] = singleRowDFT(column)
    counter = counter + 1
    
  # Set the final image straight again.
  verticalAndHorizontalDFT = verticalAndHorizontalDFT.T
  
  return verticalAndHorizontalDFT


# Start data file
fileHandle = open(OUTPUT_DIRECTORY + 'timing.csv', 'wb')
csvHandle = csv.writer(fileHandle)
csvHandle.writerow(["imageDimension", "DFT Time (sec)", "FFT Time (sec)"])

# Sleep a moment
time.sleep(2)

# Timing images of dimension 2, 4, 8, 16, ... 
for imagePower in range(1, 6):
  
  imageSize = 2 ** imagePower  
  
  # Create a random image.
  data = numpy.random.random(imageSize ** 2).reshape((imageSize, imageSize))
  
  print("Timing images of size " + str(imageSize))
  
  # Sleep a moment.
  time.sleep(1)
  
  # Time DFT
  start = time.time()
  for i in range(NUM_SAMPLES):
    myDFT(data)
  stop = time.time()
  
  # Caclulate time per run
  dftTime = (stop - start) / float(NUM_SAMPLES)
  
  # Sleep a moment.
  time.sleep(1)
  
  # Time FFT
  start = time.time()
  for i in range(NUM_SAMPLES):
    numpy.fft.fft2(data)
  stop = time.time()
  
  # Caclulate time per run
  fftTime = (stop - start) / float(NUM_SAMPLES)

  # Record it
  csvHandle.writerow([imageSize, dftTime, fftTime])

fileHandle.flush()
fileHandle.close()

print("Done.")
