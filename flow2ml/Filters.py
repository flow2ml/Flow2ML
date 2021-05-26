import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt
import random
import math

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
          plt.imsave(classPath+"/MedianImages/median"+image, cv2.cvtColor(median, cv2.COLOR_RGB2BGR))
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
      os.mkdir(classPath+"/BilateralImages")    
    
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
          cv2.imwrite(classPath+"/BilateralImages/bilateral"+image, bilateral)
        except Exception as e:
          print(f"Bilateral filter operation failed due to {e}")

  def applycanny(self,classPath):
    
    ''' 
      Applies Canny Edge Detection Filter to all the images in the given folder. 
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''

    try:
      os.mkdir(classPath+"/CannyImages")    
    except:
      raise Exception("Unable to create directory for Canny Edge Detected images.")

    for image in list(os.listdir(classPath)):
      # Read image in GRAYSCALE mode.
      img = cv2.imread(classPath+"/"+image,0)

      if img is not None:
        try:
          # applying canny edge detection filter
          cannyImage = cv2.Canny(img,self.filters['canny']['threshold_1'],self.filters['canny']['threshold_2'])
          
          # saving the image
          plt.imsave(classPath+"/CannyImages/cannyImage"+image, cannyImage)
        except Exception as e:
          print(f"Canny Edge Detection filter operation failed due to {e}")

  def visualizeFilters(self):
    ''' filtered_image
      visualizes various filtered outputs 
    '''

    # get a list of paths of all images provided
    all_image_paths = []
    for folder in self.classes:
      for i in os.listdir(os.path.join(self.dataset_dir, self.data_dir, folder)):
        if i.find("Images") == -1:
          all_image_paths.append(os.path.join(self.dataset_dir, self.data_dir, folder, i))

    # pick a random image
    random_image_path = random.choice(all_image_paths)
    head, tail = os.path.split(random_image_path)

    # get paths to all filtered images of the random image
    filtered_image_paths = [random_image_path]
    for filter in self.filters:
      filter_folder = filter.title() + "Images"
      filtered_image_paths.append(os.path.join(head, filter_folder, filter + tail))

    # create an empty plot of the required shape
    cols = 3
    rows = math.ceil(len(filtered_image_paths) / cols)
    axes = []
    fig = plt.figure()

    try:
      # add images to plot one by one
      for a in range(len(filtered_image_paths)):
        b = plt.imread(filtered_image_paths[a])
        axes.append(fig.add_subplot(rows, cols, a + 1) )
        subplot_title = os.path.split(filtered_image_paths[a])[-1]
        axes[-1].set_title(subplot_title)  
        axes[-1].set_xticks([])
        axes[-1].set_yticks([])
        plt.imshow(b)
      fig.tight_layout()    
      plt.savefig(os.path.join(self.results_path, "visualise_filters.png"))
    
    except Exception as e:
      print("Unable to create visualise_filters plot due to {e}")