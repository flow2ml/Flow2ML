import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt

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
      operation = self.operations['flip']
      if img is not None:
        # applies Flip augmentation to the image.
        Flipped = cv2.flip(img, operation)
        
        # saving the image by
        plt.imsave(classPath+"/FlippedImages/Flipped"+image, cv2.cvtColor(Flipped, cv2.COLOR_RGB2BGR))
