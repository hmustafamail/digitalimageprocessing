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
# Discrete Fourier Transform functions
#==============================================================================

def singleRowDFT(row):
  """
  Computes the Discrete Fourier Transform of a 1-D row of numbers.
  """
  
  # Thanks nayuki.io/how-to-implement-the-discrete-fourier-transform
  X = [numpy.complex(0)] * len(row)
  
  n = float(len(row))
  
  for k in range(len(row)):
    currentSum = numpy.complex(0)
    
    for t in range(len(row)):
      currentOperand = row[t] * numpy.exp((-2j * math.pi * t * k) / n)
      currentSum = currentSum + currentOperand
      
    X[k] = currentSum
    
  return X

def myDFT(image):
  """
  Computes the Discrete Fourier Transform of an image.
  """
  
  width, height = image.shape
  
  horizontalDFT = numpy.asarray([[numpy.complex(0)] * width] * height)
  verticalAndHorizontalDFT = copy.deepcopy(horizontalDFT)
  
  # First, do the rows.
  counter = 0
  for row in image:
    print("\t" + "Row " + str(counter + 1) + " of " + str(image.shape[0]) + "...")
    horizontalDFT[counter] = singleRowDFT(row)
    counter = counter + 1
    
  # Next, do the columns.
  counter = 0
  for column in horizontalDFT.T:
    print("\t" + "Column " + str(counter + 1) + " of " + str(image.shape[1]) + "...")
    verticalAndHorizontalDFT.T[counter] = singleRowDFT(column)
    counter = counter + 1
    
  return verticalAndHorizontalDFT

#==============================================================================
# Visibility functions
#==============================================================================

def visibleFFTMagnitudes(fourierImage):
  """Turns an FFT's 2D complex array into magnitudes you can see.
  Thanks to docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftn.html
  """
  return isolateMagnitude(fourierImage) ** 2

def visibleFFTPhaseAngles(fourierImage):
  """Turns an FFT's 2D complex array into phase angles you can see.
  """
  # 255 (highest intensity) / (2 * Pi (maximum radian)) = 40.585
  # I add 128 to center the angles around 128, instead of zero, so there are 
  # no negative (invisible) pixels!
  return (isolatePhaseAngle(fourierImage) * 40.585) + 128



#==============================================================================
# Isolation functions
#==============================================================================

def isolatePhaseAngle(fourierImage):
  """
  Shifts the zero-frequency to the middle and takes the element-wise
  angle of each complex number. 
  """
  return numpy.angle(numpy.fft.fftshift(fourierImage))

def isolateMagnitude(fourierImage):
  """
  Shifts the zero-frequency to the middle and takes the element-wise
  hypotenuse of each complex number. 
  
  Thanks to docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftn.html
  """
  return numpy.abs(numpy.fft.fftshift(fourierImage))



#==============================================================================
# Image reconstruction functions
#==============================================================================

def reconstructFromMagnitude(fourierImage):
  return numpy.abs(numpy.fft.ifft2(isolateMagnitude(fourierImage)))
  
def reconstructFromPhaseAngle(fourierImage):
  # 10,000 just gets it bright.
  return numpy.abs(numpy.fft.ifft2(isolatePhaseAngle(fourierImage))) * 1e4



def reconstructWithIFT(fourierImageWithMagnitudes, fourierImageWithAngles = None):
  """Reconstructs an image from the magnitudes and angles given in the input images.
  Optionally, just give one image for both magnitude and angle.
  """
  if fourierImageWithAngles == None:
    fourierImageWithAngles = fourierImageWithMagnitudes

  magnitudes = isolateMagnitude(fourierImageWithMagnitudes)
  angles = isolatePhaseAngle(fourierImageWithAngles)  

  # Convert from polar to rectangular complex coordinates  
  # Thanks to snackoverflow.com/questions/16444719
  rectangular = magnitudes * numpy.exp(1j * angles)
  
  # Inverse fourier transform it back into an image, discard erroroneous j-components.
  reconstructed = numpy.abs(numpy.fft.ifft2(rectangular))
  
  return reconstructed



#==============================================================================
# Image import/export
#==============================================================================

def saveImage(image, filename, filetype = IMAGE_FILE_EXTENSION):
  """Saves the image in the output directory with the filename given.
  """
  cv2.imwrite(OUTPUT_DIRECTORY + filename + filetype, image)
  #scipy.misc.imsave(OUTPUT_DIRECTORY + filename + IMAGE_FILE_EXTENSION, image)

def openImage(fileName):
  """Opens the image in the input directory with the filename given.
  """
  return cv2.imread(INPUT_DIRECTORY + fileName + IMAGE_FILE_EXTENSION, 0)

  
# Input images
image1Name = 'hawaii'
image2Name = 'basketball'
tinyImageName = 'fabioTiny'

# Import images.
image1 = openImage(image1Name)
image2 = openImage(image2Name)
imageForDFT = openImage(tinyImageName)


# Process images, save.
print("Performing My DFT...")
dftTiny = myDFT(imageForDFT)
dftTinyVisible = visibleFFTMagnitudes(dftTiny)
saveImage(dftTinyVisible, tinyImageName + 'DFT', ".PNG")

# FFT
print("Performing FFT...")
fftTiny = numpy.fft.fft2(imageForDFT)
fftTinyVisible = visibleFFTMagnitudes(fftTiny)
saveImage(fftTinyVisible, tinyImageName + 'FFT', ".PNG")

# Comparison
print("Subtracting DFT from FFT...")
saveImage( numpy.abs(fftTinyVisible - dftTinyVisible) , tinyImageName + 'DFTMinusFFT')

print("Plotting Spectrums...")
fftImage1 = numpy.fft.fft2(image1)
fftImage2 = numpy.fft.fft2(image2)
fftImage1Visible = visibleFFTMagnitudes(fftImage1)
fftImage2Visible = visibleFFTMagnitudes(fftImage2)
saveImage(fftImage1Visible, image1Name + 'Spectrum')
saveImage(fftImage2Visible, image2Name + 'Spectrum')

print("Plotting Phase Angles...")
fftImage1Visible = visibleFFTPhaseAngles(fftImage1)
saveImage(visibleFFTPhaseAngles(fftImage1), image1Name + 'PhaseAngle')
saveImage(visibleFFTPhaseAngles(fftImage2), image2Name + 'PhaseAngle')

print("Reconstructing via Inverse Fourier Transform...")
saveImage(reconstructWithIFT(fftImage1), image1Name + 'ReconstructedIFT')
saveImage(reconstructWithIFT(fftImage2), image2Name + 'ReconstructedIFT')

print("Reconstructing from Spectrum...")
saveImage(reconstructFromMagnitude(fftImage1), image1Name + 'ReconstructedFromSpectrum')
saveImage(reconstructFromMagnitude(fftImage2), image2Name + 'ReconstructedFromSpectrum')

print("Reconstructing from Phase Angle...")
saveImage(reconstructFromPhaseAngle(fftImage1), image1Name + 'ReconstructedFromPhaseAngle')
saveImage(reconstructFromPhaseAngle(fftImage2), image2Name + 'ReconstructedFromPhaseAngle')

print("Constructing Frankensteins...")
saveImage(reconstructWithIFT(fftImage1, fftImage2), image1Name + 'Spectrum' + image2Name + 'PhaseAngle')
saveImage(reconstructWithIFT(fftImage2, fftImage1), image2Name + 'Spectrum' + image1Name + 'PhaseAngle')

print("Done.")
