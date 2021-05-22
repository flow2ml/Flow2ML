import cv2
import os 
import imutils
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tf
from matplotlib.transforms import Affine2D
import random

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

  def applyInvert(self,classPath):
    '''
      Applies invert augmentation to all the images in the given folder. It negates the images by reversing the pixel value.
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/InvertedImages")    
    except:
      raise Exception("Unable to create directory for inverted images.")
    
    for image in list(os.listdir(classPath)):
      # Read image
      img = cv2.imread(classPath+"/"+image)
      if (self.operations['invert']==True):
        if img is not None:
          try:
            # applies Invert augmentation to the image.
            Inverted=abs(255-img)
            # saving the image by
            plt.imsave(classPath+"/InvertedImages/Inverted"+image, Inverted)
          except Exception as e:
            print(f"Inverting operation failed due to {e}")

  def applyCLAHE(self,classPath):
    '''
      Applies contrast limited adaptive histogram equalization to all the images in the given folder.
      Args:
        classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/CLAHEedImages")    
    except:
      raise Exception("Unable to create directory for CLAHEed images.")

    for image in list(os.listdir(classPath)):
      # Read image
      img = cv2.imread(classPath+"/"+image)
      if self.operations['CLAHE']==True:
        if img is not None:
          try:
            #convert BGR to GRAYSCALE
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # applies CLAHE augmentation to the image.
            clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            CLAHEed=clahe.apply(gray)
            # saving the image by
            plt.imsave(classPath+"/CLAHEedImages/CLAHEed"+image, cv2.cvtColor(CLAHEed, cv2.COLOR_GRAY2BGR))
          except Exception as e:
            print(f"CLAHE operation failed due to {e}")

  def HistogramEqualisation(self,classPath):
    '''
    Applies histogram equilisation to all the images in the given folder.It adjusts the contrast of image using the image's histogram.
    Args : 
      classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/HistogramEqualisedImages")    
    except:
      raise Exception("Unable to create directory for Histogram Equalised images.")
    
    for image in list(os.listdir(classPath)):
      # Read image
      img = cv2.imread(classPath+"/"+image)
      
      if(self.operations['Hist_Equal']==True):
        if img is not None:
          try:
            #convert image from BGR to GRAYSCALE.
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # applies histogram equalisation to the image.
            equalised_img=cv2.equalizeHist(gray)
            HistogramEqualised=np.hstack((img,equalised_img))
            # saving the image by
            plt.imsave(classPath+"/HistogramEqualisedImages/HistogramEqualised"+image, cv2.cvtColor(HistogramEqualised, cv2.COLOR_RGB2BGR))
          except Exception as e:
            print(f"Histogram Equalisation operation failed due to {e}")
    
  def applyCrop(self,classPath):
    ''' 
      Applies cropping augmentation to all the images in the given folder. Either the images are cropped randomly or cropped with fixed coordinates (y1, y2, x1, x2) given by the user
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/CroppedImages")    
    except:
      raise Exception("Unable to create directory for cropped images.")
    for image in list(os.listdir(classPath)):
      # Read image
      img = cv2.imread(classPath+"/"+image)
      if img is not None:
        try:
          if isinstance(self.operations['crop'], str):
            if self.operations['crop'] == 'random':
              y1, y2, x1, x2 = random.randint(1, img.shape[0]), random.randint(1, img.shape[0]), random.randint(1, img.shape[1]), random.randint(1, img.shape[1]),
              Cropped = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2), :]
              plt.imsave(classPath+"/CroppedImages/Cropped"+image, cv2.cvtColor(Cropped, cv2.COLOR_RGB2BGR))
          elif isinstance(self.operations['crop'], list):
            if len(self.operations['crop']) == 4:
              Cropped = img[self.operations['crop'][0]:self.operations['crop'][1], self.operations['crop'][2]:self.operations['crop'][3], :]
              plt.imsave(classPath+"/CroppedImages/Cropped"+image, cv2.cvtColor(Cropped, cv2.COLOR_RGB2BGR))
            else:
              raise Exception("Cropping needs exactly 4 coordinates for y1, y2, x1, x2.")
          else:
            raise Exception("Cropping needs random parameter or list of coordinates.")
        except Exception as e:
          print(f"Crop operation failed due to {e}")
  
  def applyScale(self,classPath):
    ''' 
      Applies scaling augmentation to all the images in the given folder. Scales the image by the given ratio.
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    ratio = self.operations['scale']
    try:
      os.mkdir(classPath+"/ScaledImages")    
    except:
      raise Exception("Unable to create directory for scaled images.")
    for image in list(os.listdir(classPath)):
      
      # Read image
      img = cv2.imread(classPath+"/"+image)

      if isinstance(self.operations['scale'], str):
        raise Exception("Scaling ratio cannot be a string.")
      else:
        if img is not None:
          try:
            if ratio < 0:
              raise Exception("Scale ratio cannot be negative.")
            else:
              # applies scale augmentation to the image.
              Scaled = cv2.resize(img, (round(img.shape[0] * ratio), round(img.shape[1] * ratio)))
              # # saving the image by
              plt.imsave(classPath+"/ScaledImages/Scaled"+image, cv2.cvtColor(Scaled, cv2.COLOR_RGB2BGR))
          except Exception as e:
            print(f"Scale operation failed due to {e}")

  def applyZoom(self,classPath):
    ''' 
      Applies zooming augmentation to all the images in the given folder. It zooms the images by the given ratio
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/ZoomedImages")    
    except:
      raise Exception("Unable to create directory for zoomed images.")

    for image in list(os.listdir(classPath)):
      
      # Read image
      img = cv2.imread(classPath+"/"+image)
      
      if isinstance(self.operations['zoom'], str):
        raise Exception("Zoom factor cannot be a string.")
      else:
        
        factor = self.operations['zoom']
        if factor < 1:
          raise Exception("Zoom factor cannot be lesser than 1.")
        else:
          if img is not None:
            try:
              # applies zooming augmentation to the image.
              h, w = img.shape[0], img.shape[1]
              Zoomed = cv2.resize(img, (round(img.shape[1] * factor), round(img.shape[0] * factor)))
              w_zoomed, h_zoomed = Zoomed.shape[1], Zoomed.shape[0]
              x1 = round((float(w_zoomed) / 2) - (float(w) / 2))
              x2 = round((float(w_zoomed) / 2) + (float(w) / 2))
              y1 = round((float(h_zoomed) / 2) - (float(h) / 2))
              y2 = round((float(h_zoomed) / 2) + (float(h) / 2))
              Zoomed = Zoomed[y1:y2, x1:x2]         
              # saving the image by
              plt.imsave(classPath+"/ZoomedImages/Zoomed"+image, cv2.cvtColor(Zoomed, cv2.COLOR_RGB2BGR))
            except Exception as e:
              print(f"Zooming operation failed due to {e}")

  def applyGreyscale(self,classPath):
    ''' 
      Applies greyscale augmentation to all the images in the given folder.
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/GreyscaleImages")    
    except:
      raise Exception("Unable to create directory for greyscale images.")

    for image in list(os.listdir(classPath)):
      
      # Read image
      img = cv2.imread(classPath+"/"+image)
      
      if not isinstance(self.operations['greyscale'], bool):
        raise Exception("Greyscale parameter must be a boolean value.")
      else:
        
        if self.operations['greyscale']:
          if img is not None:
            try:
              # applies greyscale augmentation to the image.
              Greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
              # saving the image by
              cv2.imwrite(classPath+"/GreyscaleImages/Greyscale"+image, Greyscale)
            except Exception as e:
              print(f"Greyscale operation failed due to {e}")

  def applyErode(self,classPath):
    ''' 
      Applies Erosion augmentation to all the images in the given folder.
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/ErodedImages")    
    except:
      raise Exception("Unable to create directory for eroded images.")

    for image in list(os.listdir(classPath)):
      
      # Read image
      img = cv2.imread(classPath+"/"+image)
      
      if self.operations['erode']==True:
        if img is not None:
          try:
            # apply Erosion on the image.
            kernel=np.ones((5,5),np.uint8)
            Eroded=cv2.erode(img,kernel,iterations=1) 
            # saving the image by
            cv2.imwrite(classPath+"/ErodedImages/Eroded"+image, Eroded)
          except Exception as e:
            print(f"Erosion operation failed due to {e}")

  def applyDilate(self,classPath):
    ''' 
      Applies Dilation augmentation to all the images in the given folder.
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/DilatedImages")    
    except:
      raise Exception("Unable to create directory for dilated images.")

    for image in list(os.listdir(classPath)):
      
      # Read image
      img = cv2.imread(classPath+"/"+image)
      
      if self.operations['dilate']==True:
        if img is not None:
          try:
            # apply Dilation on the image.
            kernel=np.ones((5,5),np.uint8)
            Dilated=cv2.dilate(img,kernel,iterations=1) 
            # saving the image by
            cv2.imwrite(classPath+"/DilatedImages/Dilated"+image, Dilated)
          except Exception as e:
            print(f"Dilation operation failed due to {e}")

  def applyOpen(self,classPath):
    ''' 
      Applies Opening augmentation to all the images in the given folder.
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/OpenedImages")    
    except:
      raise Exception("Unable to create directory for Opened images.")

    for image in list(os.listdir(classPath)):
      
      # Read image
      img = cv2.imread(classPath+"/"+image)
      
      if self.operations['open']==True:
        if img is not None:
          try:
            # apply Opening on the image.
            kernel=np.ones((5,5),np.uint8)
            Opened=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel) 
            # saving the image by
            cv2.imwrite(classPath+"/OpenedImages/Opened"+image, Opened)
          except Exception as e:
            print(f"Opening operation failed due to {e}")

  def applyClose(self,classPath):
    ''' 
      Applies Closing augmentation to all the images in the given folder.
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/ClosedImages")    
    except:
      raise Exception("Unable to create directory for Closed images.")

    for image in list(os.listdir(classPath)):
      
      # Read image
      img = cv2.imread(classPath+"/"+image)
      
      if self.operations['close']==True:
        if img is not None:
          try:
            # apply Closing on the image.
            kernel=np.ones((5,5),np.uint8)
            Closed=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel) 
            # saving the image by
            cv2.imwrite(classPath+"/ClosedImages/Closed"+image, Closed)
          except Exception as e:
            print(f"Closing operation failed due to {e}")

  def applyThreshold(self,classPath):
    ''' 
      Applies thresholding augmentation to all the images in the given folder.
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''

    try:
      os.mkdir(classPath+"/ThresholdedImages")    
    except:
      raise Exception("Unable to create directory for thresholded images.")
    for image in list(os.listdir(classPath)):
      
      # Read image
      img = cv2.imread(classPath+"/"+image)
      
      if self.operations['threshold'] not in ['simple', 'adaptive', 'OTSU']:
        raise Exception("Invalid threshold operation.")
      else:
        
        if self.operations['threshold'] == 'simple':
          if img is not None:
            try:
              #convert the image from BGR to GRAYSCALE
              gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
              # applies simple thresholding augmentation to the image.
              _,s_threshed=cv2.threshold(gray,120,255,cv2.THRESH_BINARY)  
              # saving the image by
              plt.imsave(classPath+"/ThresholdedImages/s_threshed"+image,s_threshed)
            except Exception as e:
              print(f"simple thresholding operation failed due to {e}")

        elif self.operations['threshold'] == 'adaptive':
          if img is not None:
            try:
              #convert the image from BGR to GRAYSCALE
              gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
              # applies adaptive thresholding augmentation to the image.
              a_threshed=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,199,5)  
              # saving the image by
              plt.imsave(classPath+"/ThresholdedImages/a_threshed"+image,a_threshed)
            except Exception as e:
              print(f"adaptive thresholding operation failed due to {e}")

        elif self.operations['threshold'] == 'OTSU':
          if img is not None:
            try:
              #convert the image from BGR to GRAYSCALE
              gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
              # applies OTSU thresholding augmentation to the image.
              _,o_threshed=cv2.threshold(gray,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
              # saving the image by
              plt.imsave(classPath+"/ThresholdedImages/o_threshed"+image,o_threshed)
            except Exception as e:
              print(f"OTSU thresholding operation failed due to {e}")
