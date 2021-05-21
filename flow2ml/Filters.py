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
    except:
      raise Exception("Unable to create directory for Median images.")

    for image in list(os.listdir(classPath)):
      
      # Read image
      img = cv2.imread(classPath+"/"+image)

      if img is not None:
        try:
          # applying median filter
          median = cv2.medianBlur(img, 5)
          
          # saving the image
          plt.imsave(classPath+"/MedianImages/Median"+image, cv2.cvtColor(median, cv2.COLOR_RGB2BGR))
        except Exception as e:
          print(f"Median filter operation failed due to {e}")
  
  def applylaplacian(self,classPath):
    ''' 
      Applies Laplacian Filter to all the images in the given folder. 
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    
    try:
      os.mkdir(classPath+"/LaplacianImages")  
    except:
      raise Exception("Unable to create directory for Laplacian images.")

    for image in list(os.listdir(classPath)):
      
      # Read image
      img = cv2.imread(classPath+"/"+image)

      if img is not None:
        try:
          # applying laplacian filter
          laplacian = cv2.Laplacian(img,cv2.CV_64F)
          
          # saving the image
          cv2.imwrite(classPath+"/LaplacianImages/laplacian"+image, laplacian)
        except Exception as e:
          print(f"Laplacian filter operation failed due to {e}")

  def applysobelx(self,classPath):
    ''' 
      Applies Sobel-x Filter to all the images in the given folder. 
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''

    try:
      os.mkdir(classPath+"/SobelxImages")    
    except:
      raise Exception("Unable to create directory for Sobelx images.")

    for image in list(os.listdir(classPath)):
      # Read image
      img = cv2.imread(classPath+"/"+image)

      if img is not None:
        try:
          # applying sobelx filter
          sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
          
          # saving the image
          cv2.imwrite(classPath+"/SobelxImages/sobelx"+image, sobelx)
        except Exception as e:
          print(f"Sobelx filter operation failed due to {e}")

  def applysobely(self,classPath):
    
    ''' 
      Applies Sobel-y Filter to all the images in the given folder. 
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/SobelyImages")    
    except:
      raise Exception("Unable to create directory for Sobely images.")

    for image in list(os.listdir(classPath)):
      # Read image
      img = cv2.imread(classPath+"/"+image)

      if img is not None:
        try:
          # applying sobely filter
          sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
          
          # saving the image
          cv2.imwrite(classPath+"/SobelyImages/sobely"+image, sobely)
        except Exception as e:
          print(f"Sobely filter operation failed due to {e}")

  def applygaussian(self,classPath):
    
    ''' 
      Applies gaussian Filter to all the images in the given folder. 
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''

    try:
      os.mkdir(classPath+"/GaussianImages")    
    except:
      raise Exception("Unable to create directory for Gaussian images.")

    for image in list(os.listdir(classPath)):
      # Read image
      img = cv2.imread(classPath+"/"+image)

      if img is not None:
        try:
          # applying gaussian filter
          gaussian = cv2.GaussianBlur(img,(5,5),0)
          
          # saving the image
          plt.imsave(classPath+"/GaussianImages/gaussian"+image, cv2.cvtColor(gaussian, cv2.COLOR_RGB2BGR))
        except Exception as e:
          print(f"Gaussian filter operation failed due to {e}")
        
  def applybilateral(self,classPath):
    ''' 
      Applies Bilateral Filter to all the images in the given folder. 
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/Bilaterally_FilteredImages")    
    
    except:
      raise Exception("Unable to create directory for bilateral images.")
      

    for image in list(os.listdir(classPath)):
      # Read image
      img = cv2.imread(classPath+"/"+image)

      if img is not None:
        try:
          # applying bilateral filter
          bilateral = cv2.bilateralFilter(img,15,80,80)
          
          # saving the image
          cv2.imwrite(classPath+"/Bilaterally_FilteredImages/bilateral"+image, bilateral)
        except Exception as e:
          print(f"Bilateral filter operation failed due to {e}")

  def visualizeFilters(self):
    ''' filtered_image
      visualizes various filtered outputs 
    '''

    ####### Note #######
    ''' To be completed '''
    ####### Note #######
    pass
