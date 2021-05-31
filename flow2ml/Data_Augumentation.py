import cv2
import os 
import imutils
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tf
from matplotlib.transforms import Affine2D
import random
import math

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
      
      if self.operations['flipped'] not in ['horizontal', 'vertical', 'cross']:
        raise Exception("Invalid flip operation.")
      else:
        
        if self.operations['flipped'] == 'horizontal':
          operation = 1
        elif self.operations['flipped'] == 'vertical':
          operation = 0
        elif self.operations['flipped'] == 'cross':
          operation = -1
        if img is not None:
          
          try:
            # applying flip augmentation to the image.
            Flipped = cv2.flip(img, operation)  
            # saving the image
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
      
      if isinstance(self.operations['rotated'], str):
        raise Exception("Rotation angle cannot be a string.")
      else:
        # get absolute value of the angle
        angle = round(self.operations['rotated']) % 360
        if img is not None:
          try:
            # applying rotate augmentation to the image.
            Rotated = imutils.rotate(img, angle)            
            # saving the image
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
      if isinstance(self.operations['sheared']['x_axis'], str) or isinstance(self.operations['sheared']['y_axis'], str):
        raise Exception("Shearing angle cannot be a string.")
      else:
        angle_x = np.deg2rad(self.operations['sheared']['x_axis'])
        angle_y = np.deg2rad(self.operations['sheared']['y_axis'])
        if img is not None:
          try:
            # applying shear augmentation to the image.
            Sheared = tf.warp(img, inverse_map = np.linalg.inv(Affine2D().skew(xShear = angle_y, yShear = angle_y).get_matrix()))
            Sheared = (Sheared * 255).astype(np.uint8) 
            # saving the image
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
      if (self.operations['inverted']==True):
        if img is not None:
          try:
            # applying invert augmentation to the image.
            Inverted=abs(255-img)
            # saving the image
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
      os.mkdir(classPath+"/CLAHEImages")    
    except:
      raise Exception("Unable to create directory for CLAHE images.")

    for image in list(os.listdir(classPath)):
      # Read image
      img = cv2.imread(classPath+"/"+image)
      if self.operations['CLAHE']==True:
        if img is not None:
          try:
            # convert BGR to GRAYSCALE
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # applying CLAHE augmentation to the image.
            clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            CLAHE=clahe.apply(gray)
            # saving the image
            plt.imsave(classPath+"/CLAHEImages/CLAHE"+image, cv2.cvtColor(CLAHE, cv2.COLOR_GRAY2BGR))
          except Exception as e:
            print(f"CLAHE operation failed due to {e}")

  def applyHistogramEqualization(self,classPath):
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
      
      if(self.operations['histogramequalised']==True):
        if img is not None:
          try:
            # convert from RGB color-space to YCrCb
            ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            # equalize the histogram of the Y channel
            ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
            # convert back to RGB color-space from YCrCb
            equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
            plt.imsave(classPath+"/HistogramEqualisedImages/HistogramEqualised"+image, cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB))
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
          if isinstance(self.operations['cropped'], str):
            # generate random coordinates and crop the image
            if self.operations['cropped'] == 'random':
              y1, y2, x1, x2 = random.randint(1, img.shape[0]), random.randint(1, img.shape[0]), random.randint(1, img.shape[1]), random.randint(1, img.shape[1]),
              Cropped = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2), :]
              plt.imsave(classPath+"/CroppedImages/Cropped"+image, cv2.cvtColor(Cropped, cv2.COLOR_RGB2BGR))
          elif isinstance(self.operations['cropped'], list):
            if len(self.operations['cropped']) == 4:
              # crop image by given coordinates only
              Cropped = img[self.operations['cropped'][0]:self.operations['cropped'][1], self.operations['cropped'][2]:self.operations['cropped'][3], :]
              # saving the image
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
    ratio = self.operations['scaled']
    try:
      os.mkdir(classPath+"/ScaledImages")    
    except:
      raise Exception("Unable to create directory for scaled images.")
    for image in list(os.listdir(classPath)):
      
      # Read image
      img = cv2.imread(classPath+"/"+image)

      if isinstance(self.operations['scaled'], str):
        raise Exception("Scaling ratio cannot be a string.")
      else:
        if img is not None:
          try:
            if ratio < 0:
              raise Exception("Scale ratio cannot be negative.")
            else:
              # applying scale augmentation to the image.
              Scaled = cv2.resize(img, (round(img.shape[0] * ratio), round(img.shape[1] * ratio)))
              # saving the image
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
      
      if isinstance(self.operations['zoomed'], str):
        raise Exception("Zoom factor cannot be a string.")
      else:
        
        factor = self.operations['zoomed']
        if factor < 1:
          raise Exception("Zoom factor cannot be lesser than 1.")
        else:
          if img is not None:
            try:
              # applying zooming augmentation to the image.
              h, w = img.shape[0], img.shape[1]
              # scale the image by the given factor
              Zoomed = cv2.resize(img, (round(img.shape[1] * factor), round(img.shape[0] * factor)))
              w_zoomed, h_zoomed = Zoomed.shape[1], Zoomed.shape[0]
              # crop the middle of the image by original dimensions
              x1 = round((float(w_zoomed) / 2) - (float(w) / 2))
              x2 = round((float(w_zoomed) / 2) + (float(w) / 2))
              y1 = round((float(h_zoomed) / 2) - (float(h) / 2))
              y2 = round((float(h_zoomed) / 2) + (float(h) / 2))
              Zoomed = Zoomed[y1:y2, x1:x2]         
              # saving the image
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
              # applying greyscale augmentation to the image.
              Greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
              # saving the image
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
      
      if self.operations['eroded']==True:
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
      
      if self.operations['dilated']==True:
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
      
      if self.operations['opened']==True:
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
      
      if self.operations['closed']==True:
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
      if self.operations['thresholded']['type'] not in ['simple', 'adaptive', 'OTSU']:
        raise Exception("Invalid operation.")
      else:
        if img is not None:
          if self.operations['thresholded']['type'] == 'adaptive':
            try:
              #convert the image from BGR to GRAYSCALE
              gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
              # applies adaptive thresholding augmentation to the image.
              a_threshed=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,199,5)  
              # saving the image by
              plt.imsave(classPath+"/ThresholdedImages/thresholded"+image, cv2.cvtColor(a_threshed, cv2.COLOR_GRAY2BGR))
            except Exception as e:
              print(img)
              print(f"Adaptive thresholding operation failed due to {e}")

          elif isinstance(self.operations['thresholded']['thresh_val'],str):
            raise Exception("Threshold value can't be a string.")
          else:
            Threshold=self.operations['thresholded']['thresh_val']

            if(self.operations['thresholded']['type']== 'OTSU'):
                try:
                  #convert the image from BGR to GRAYSCALE
                  gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                  # applies OTSU thresholding augmentation to the image.
                  _,o_threshed=cv2.threshold(gray,Threshold,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
                  # saving the image by
                  plt.imsave(classPath+"/ThresholdedImages/thresholded"+image, cv2.cvtColor(o_threshed, cv2.COLOR_GRAY2BGR))
                except Exception as e:
                  print(f"OTSU thresholding operation failed due to {e}") 

            elif(self.operations['thresholded']['type']=='simple'):
                try:
                  #convert the image from BGR to GRAYSCALE
                  gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                  # applies simple thresholding augmentation to the image.
                  _,s_threshed=cv2.threshold(gray,Threshold,255,cv2.THRESH_BINARY)  
                  # saving the image by
                  plt.imsave(classPath+"/ThresholdedImages/thresholded"+image, cv2.cvtColor(s_threshed, cv2.COLOR_GRAY2BGR))
                except Exception as e:
                  print(f"Simple thresholding operation failed due to {e}")

  def changeColorSpace(self,classPath):
    ''' 
      Applies changing color-space augmentation to all the images in the given folder.
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''
    try:
      os.mkdir(classPath+"/ColorspaceImages")    
    except:
      raise Exception("Unable to create directory for changed images.")
    for image in list(os.listdir(classPath)):

      if((self.operations['colorspace']['input'] not in ['BGR','GRAY']) or (self.operations['colorspace']['output'] not in ['BGR','RGB','GRAY'])):
        raise Exception("Invalid colorspace operation.")
      else:

        if(self.operations['colorspace']['input']=='BGR'):

          #Read image in BGR mode.
          img=cv2.imread(classPath+"/"+image)
          if img is not None:
            if (self.operations['colorspace']['output']=='RGB'):
              try:
                # changing color-space of the image from BGR to RGB.
                changed = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  
                # saving the image
                plt.imsave(classPath+"/ColorspaceImages/Colorspace"+image, changed)
              except Exception as e:
                print(f"color-space operation failed due to {e}")

        elif(self.operations['colorspace']['input']=='GRAY'):

          #Read image in grayscale mode.
          img=cv2.imread(classPath+"/"+image,0)
          if img is not None:
            if (self.operations['colorspace']['output']=='RGB'):
              try:
                # changing color-space of the image from GRAY to RGB.
                changed = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)  
                # saving the image
                plt.imsave(classPath+"/ColorspaceImages/Colorspace"+image, changed)
              except Exception as e:
                print(f"color-space operation failed due to {e}")

            if (self.operations['colorspace']['output']=='BGR'):
              try:
                # changing color-space of the image from GRAY to BGR.
                changed = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)  
                # saving the image
                plt.imsave(classPath+"/ColorspaceImages/Colorspace"+image, changed)
              except Exception as e:
                print(f"color-space operation failed due to {e}")

  def applyCanny(self,classPath):
    
    ''' 
      Applies Canny Edge Detection to all the images in the given folder. 
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
          # applying canny edge detection
          cannyImage = cv2.Canny(img,self.operations['canny']['threshold_1'],self.operations['canny']['threshold_2'])
          
          # saving the image
          plt.imsave(classPath+"/CannyImages/Canny"+image, cv2.cvtColor(cannyImage, cv2.COLOR_GRAY2RGB))
        except Exception as e:
          print(f"Canny Edge Detection operation failed due to {e}")

  def applyEnhanceBrightness(self,classPath):
    
    ''' 
      Applies log transformation to enhance brightness in all the images in the given folder. 
      Args : 
        classPath : (string) directory containing images for a particular class.
    '''

    try:
      os.mkdir(classPath+"/BrightnessEnhancedImages")    
    except:
      raise Exception("Unable to create directory for brightness enhanced images.")

    for image in list(os.listdir(classPath)):

      img = cv2.imread(classPath+"/"+image)

      if img is not None:
        try:
          # calculate and apply log transform
          c = 255 / np.log(1 + np.max(img))
          brightnessEnhancedImage = c * (np.log(img + 1e-7))
          brightnessEnhancedImage = np.array(brightnessEnhancedImage, dtype = np.uint8)

          # saving the image
          plt.imsave(classPath+"/BrightnessEnhancedImages/BrightnessEnhanced"+image, cv2.cvtColor(brightnessEnhancedImage, cv2.COLOR_BGR2RGB))
        except Exception as e:
          print(f"Brightness enhancement operation failed due to {e}")

  def visualizeAugmentation(self):

    ''' 
      Visualises all given augmentations for a randomly picked image. 
      Args : None.
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

    # get paths to all augmentated images of the random image
    augmented_image_paths = [random_image_path]
    for operation in self.operations:
      augmented_folder = operation.title() + "Images"
      augmented_image_paths.append(os.path.join(head, augmented_folder, operation + tail))

    # create an empty plot of the required shape
    cols = 4
    rows = math.ceil(len(augmented_image_paths) / cols)
    axes = []
    fig = plt.figure()

    try:
      # add images to plot one by one
      for a in range(len(augmented_image_paths)):
        b = plt.imread(augmented_image_paths[a])
        axes.append(fig.add_subplot(rows, cols, a + 1) )
        subplot_title = os.path.split(augmented_image_paths[a])[-1]
        axes[-1].set_title(subplot_title, fontsize = 7.5)  
        axes[-1].set_xticks([])
        axes[-1].set_yticks([])
        plt.imshow(b)
      fig.tight_layout()    
      plt.savefig(os.path.join(self.results_path, "visualise_augmentation.png"))
    
    except Exception as e:
      print(f"Unable to create visualise_augmentation plot due to {e}")