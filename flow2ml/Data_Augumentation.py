import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt

class Data_Augumentation:
  '''
    Class containing methods to apply Data Augumentation techniques
    to images in the data folder  
  '''

  def __init__(self,techniques):
    '''
      Initializes various attributes regarding to the object.
      Args : 
        techniques : (dictonary) python dictonary containing key value pairs
                      of techniques and values to be applied to the image data.
    '''
    self.techniques = techniques

  def applyFlip(self,classPath):
    ''' 
      Applies flipping augmentation to all the images in the given folder. 
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/FlippedImages")    
    except:
      pass

    for image in list(os.listdir(classPath)):
      # Read image
      img = cv2.imread(classPath+"/"+image)
      technique = self.techniques['flip']
      if img is not None:
        # applies Flip augmentation to the image.
        Flipped = cv2.flip(img, technique)
        
        # saving the image by
        plt.imsave(classPath+"/FlippedImages/Flipped"+image, cv2.cvtColor(Flipped, cv2.COLOR_RGB2BGR))
