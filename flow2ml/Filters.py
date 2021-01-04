import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt

class Filters:
  '''
    Class containing methods to apply filters to images in the data folder  
  '''

  def __init__(self,filters):
    '''
      Initializes various attributes regarding to the object.
      Args : 
        filters : (List) python list containing various filters to 
                         be applied to the image data.
    '''
    self.filters = filters


  def applyMedian(self,classPath):
    ''' 
      Applies Median Filter to all the images in the given folder. 
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/MedianImages")    
      print("Creating MedianImages Folder")
    except:
      pass

    for image in list(os.listdir(classPath)):
      # Read image
      img = cv2.imread(classPath+"/"+image)

      if img is not None:
        # applies median filter to the image.
        median = cv2.medianBlur(img, 5)
        
        # saving the image by
        plt.imsave(classPath+"/MedianImages/Median"+image, median)
  
  def applylaplacian(self,classPath):
    ''' 
      Applies Laplacian Filter to all the images in the given folder. 
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    
    try:
      os.mkdir(classPath+"/LaplacianImages")    
      print("Creating LaplacianImages Folder")
    except:
      pass

    for image in list(os.listdir(classPath)):
      # Read image
      img = cv2.imread(classPath+"/"+image)

      if img is not None:
        # applying laplacian filter
        laplacian = cv2.Laplacian(img,cv2.CV_64F)
        
        # saving the image.
        cv2.imwrite(classPath+"/LaplacianImages/laplacian"+image, laplacian)

  def applysobelx(self,classPath):
    ''' 
      Applies Sobel-x Filter to all the images in the given folder. 
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''

    try:
      os.mkdir(classPath+"/SobelxImages")    
      print("Creating SobelxImages Folder")
    except:
      pass

    for image in list(os.listdir(classPath)):
      # Read image
      img = cv2.imread(classPath+"/"+image)

      if img is not None:
        # applying sobelx filter
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        
        # saving the image.
        cv2.imwrite(classPath+"/SobelxImages/sobelx"+image, sobelx)

  def applysobely(self,classPath):
    ''' 
      Applies Sobel-y Filter to all the images in the given folder. 
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/SobelyImages")    
      print("Creating SobelyImages Folder")
    except:
      pass

    for image in list(os.listdir(classPath)):
      # Read image
      img = cv2.imread(classPath+"/"+image)

      if img is not None:
        # applying Sobel-y filter
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
        
        # saving the image.
        cv2.imwrite(classPath+"/SobelyImages/sobely"+image, sobely)


  def applygaussian(self,classPath):
    ''' 
      Applies gaussian Filter to all the images in the given folder. 
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''

    try:
      os.mkdir(classPath+"/GaussianImages")    
      print("Creating GaussianImages Folder")
    except:
      pass

    for image in list(os.listdir(classPath)):
      # Read image
      img = cv2.imread(classPath+"/"+image)

      if img is not None:
        # applying gaussian filter
        gaussian = cv2.GaussianBlur(img,(5,5),0)
        
        # saving the image.
        plt.imsave(classPath+"/GaussianImages/gaussian"+image, gaussian)

  def visualizeFilters(self):
    ''' filtered_image
      visualizes various filtered outputs 
    '''

    ####### Note #######
    ''' To be completed '''
    ####### Note #######
    pass