import cv2
import os 
import imutils
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tf
from matplotlib.transforms import Affine2D

class Data_Augumentation:
  '''
    Class containing methods to apply Data Augumentation operations
    to images in the data folder  
  '''

  def __init__(self,operations):
    '''
      Initializes various attributes regarding to the object.
      Args : 
        operations : (dictionary) python dictionary containing key value pairs
                      of operations (flip, shear, zoom etc.) and values (integer) to be applied to the image data.
    '''
    self.operations = operations

  def applyFlip(self,classPath):
    ''' 
      Applies flipping augmentation to all the images in the given folder. Flip: 1 flips it on y axis, 0 flips it on x axis and -1 flips it on both axis
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/FlippedImages")    
    except:
      raise Exception("Unable to create directory for flipped images.")

    for image in list(os.listdir(classPath)):
      # Read image
      img = cv2.imread(classPath+"/"+image)
      if self.operations['flip'] not in ['horizontal', 'vertical', 'cross']:
        raise Exception("Invalid flip operation.")
      else:
        if self.operations['flip'] == 'horizontal':
          operation = 1
        elif self.operations['flip'] == 'vertical':
          operation = 0
        elif self.operations['flip'] == 'cross':
          operation = -1
        if img is not None:
          try:
            # applies Flip augmentation to the image.
            Flipped = cv2.flip(img, operation)
            
            # saving the image by
            plt.imsave(classPath+"/FlippedImages/Flipped"+image, cv2.cvtColor(Flipped, cv2.COLOR_RGB2BGR))
          except Exception as e:
            print(f"Flip operation failed due to {e}")
  
  def applyRotate(self,classPath):
    ''' 
      Applies rotation augmentation to all the images in the given folder. It rotates the images by the given angle (in degrees) in the counter clockwise direction
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/RotatedImages")    
    except:
      raise Exception("Unable to create directory for rotated images.")

    for image in list(os.listdir(classPath)):
      # Read image
      img = cv2.imread(classPath+"/"+image)
      if isinstance(self.operations['rotate'], str):
        raise Exception("Rotation angle cannot be a string.")
      else:
        angle = round(self.operations['rotate']) % 360
        if img is not None:
          try:
            # applies Rotate augmentation to the image.
            Rotated = imutils.rotate(img, angle)
            
            # saving the image by
            plt.imsave(classPath+"/RotatedImages/Rotated"+image, cv2.cvtColor(Rotated, cv2.COLOR_RGB2BGR))
          except Exception as e:
            print(f"Rotation operation failed due to {e}")

  def applyShear(self,classPath):
    ''' 
      Applies shear augmentation to all the images in the given folder. It shears the images by the given angles (in degrees) along the given axes
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/ShearedImages")    
    except:
      raise Exception("Unable to create directory for sheared images.")

    for image in list(os.listdir(classPath)):
      # Read image
      img = cv2.imread(classPath+"/"+image)
      if isinstance(self.operations['shear']['x_axis'], str) or isinstance(self.operations['shear']['y_axis'], str):
        raise Exception("Shearing angle cannot be a string.")
      else:
        angle_x = np.deg2rad(self.operations['shear']['x_axis'])
        angle_y = np.deg2rad(self.operations['shear']['y_axis'])
        if img is not None:
          try:
            # applies Rotate augmentation to the image.
            Sheared = tf.warp(img, inverse_map = np.linalg.inv(Affine2D().skew(xShear = angle_y, yShear = angle_y).get_matrix()))
            Sheared = (Sheared * 255).astype(np.uint8) 
            # saving the image by
            plt.imsave(classPath+"/ShearedImages/Sheared"+image, cv2.cvtColor(Sheared, cv2.COLOR_RGB2BGR))
          except Exception as e:
            print(f"Shearing operation failed due to {e}")

  
  def HistogramEqualisation(image_file):
    '''class containing method to implement histogram equalization to a given image.'''
    #read the image and flag is 0 here,to indicate image is loaded in grayscale mode.
    img=cv2.imread(image_file,0) 
    #Equalize the Histogram for contrast adjustment.
    equalised_img=cv2.equalizeHist(img)
    #stacking both the original and equalised images.
    result=np.hstack((img,equalised_img))
    #final image is in output.png file
    cv2.imwrite("output.png",result)

  