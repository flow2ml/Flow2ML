import cv2
import os 
import shutil
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Data_Loader:
  '''
    Class containing methods to load datsets.  
  '''
  
  def __init__(self,dataset_dir,data_dir):
    '''
      Initializes various attributes regarding to the object.
      Args :
        dataset_dir : (string) the datset containing main folder.
        data_dir : (string) the data containing folder with various labels
                            as sub folders.
    '''
    self.dataset_dir = dataset_dir
    self.data_dir = data_dir
    self.img_label = {}
    self.final_data_folder = os.path.join(self.dataset_dir,'processedData')
    self.num_classes = 0
    self.classes = []
  
  def getClasses(self):
    '''
      Returns all the class names present in the main directory
      Args :
        None
      Return :
        list . A python list containing all classes
    '''
    data_folder = os.path.join(self.dataset_dir,self.data_dir)
    training_classes = [f.name for f in os.scandir(data_folder) if f.is_dir()]
    self.num_classes = len(training_classes)
    self.classes = training_classes
    return training_classes

  def process_images(self,img_source,img_dest,img_name,classPath):
    '''
      Copies the given image to the processedData folder and assigns the relevant label
      Args :
        img_source : (string) path to the image to be copied
        img_dest : (string) path to where the image should be copied
        img_name : (string) original image name
        classPath : (string) directory containing images for a particular class.
      Return :
        None
    '''
    shutil.copy(img_source,img_dest)
              
    # Handle path seperators for Linux and Windows
    try:
      if '/' in classPath:
        folderName = list(classPath.split("/"))[2]
      else:
        folderName = os.path.split(classPath)[1]
    except Exception as e:
      print(f"Unable to extract foldername due to {e}")
      
                
    # Assign relevant labels to the image
    ind = self.classes.index(folderName)
    self.img_label[img_name] = np.squeeze(np.eye(len(self.classes))[ind])

  def create_dataset(self,classPath):
    '''
      copies all the processed images to a single folder and 
      creates a dictonary with image and its ground truth value.
      Args :
        classPath : (string) directory containing images for a particular class.
      Returns :
        dictonary. A python dictonary containing imageName as key
                   and one hot encoded class name as value
    '''

    final_data_folder = os.path.join(self.dataset_dir,'processedData')

    for image in list(os.listdir(classPath)):
      
        if (image.find("Images") != -1):
          # path is a folder containing multiple preprocessed images

            for image_in_folder in (list(os.listdir(os.path.join(classPath,image)))):
              

              filtered_image = os.path.join(classPath,image,image_in_folder)
              self.process_images(filtered_image, final_data_folder, image_in_folder, classPath)
    
    # If no folders of preprocessed images are found, move original images to processedData
    if not any([i != -1 for i in [image.find("Images") for image in os.listdir(classPath)]]):

      for image in list(os.listdir(classPath)):

        if not(image.find("Images") != -1):
        # path is an image since folder could not be found
        
          original_image = os.path.join(classPath,image)
          self.process_images(original_image, final_data_folder, image, classPath)

    return self.img_label

  def resize_image(self,img_resize_shape,image_path):
    '''
      method to resize the image to the given arguments.
      Args:
        img_resize_shape: (tuple). The images are
                           resized to the given size.
        image_path : (string). image name
    '''

    image_path = os.path.join(self.final_data_folder,image_path)

    # Read Image
    img = plt.imread(image_path) 

    # Resize Image
    img = cv2.resize(img,img_resize_shape)
    
    # return resized image
    return img

  def prepare_dataset(self,img_resize_shape,train_val_split,img_label_dict,random_state,encoding):
    '''
      Resizes the dataset and Prepares train, validation sets.
      Args :
         img_resize_shape: (tuple). The images are resized to the given size.

         train_val_split: (float). value used to split the entire dataset
                                   to train and validation sets.

         img_label_dict: (dictonary). contains image name as key and its 
                                      label as value  

    '''

    self.img_label = img_label_dict
    # differentiating the complete dataset into training and validating datasets.
    img_name_train, img_name_val, output_label_train, output_label_val = train_test_split(
                                                                    list(self.img_label.keys()),
                                                                    list(self.img_label.values()),
                                                                    test_size=train_val_split,
                                                                    random_state=random_state)
    

    # Creating Numpy Dataset
    ( Height, Width, channels ) = img_resize_shape
    img_height_width = (Height, Width)
    
    train_images = np.ndarray(shape=(len(img_name_train), Height, Width, channels), dtype=np.float32)
    train_labels = np.ndarray(shape=(len(output_label_train), self.num_classes ), dtype=np.float32)
    val_images = np.ndarray(shape=(len(img_name_val), Height, Width, channels), dtype=np.float32)
    val_labels = np.ndarray(shape=(len(output_label_val), self.num_classes ), dtype=np.float32)
    # Adding Values to the numpy datasets
    i=0
    for image in list(img_name_train):
      x = self.resize_image(img_height_width,image)
      # Create an empty array of required shape, then copy only the required number of channels from image
      # Works for discarding alpha channel if image dimensions specify only 3 channels
      temp = np.zeros(img_resize_shape)
      for j in range(channels):
        temp[:,:,j] = x[:,:,j]
      train_images[i] = temp
      train_labels[i] = np.asarray(output_label_train[i])
      i += 1


    i=0
    for image in list(img_name_val):
      x = self.resize_image(img_height_width,image)
      # Create an empty array of required shape, then copy only the required number of channels from image
      # Works for discarding alpha channel if image dimensions specify only 3 channels
      temp = np.zeros(img_resize_shape)
      for j in range(channels):
        temp[:,:,j] = x[:,:,j]
      val_images[i] = temp
      val_labels[i] = np.asarray(output_label_val[i])
      i += 1

    # default encoding is one hot encoding
    if encoding == 'one-hot':
      pass
    elif encoding == 'label':
      train_labels = np.array([np.argmax(i) for i in train_labels])
      val_labels = np.array([np.argmax(i) for i in val_labels])
    else:
      raise Exception(f"Not a valid option for encoding.")
    return (train_images,train_labels,val_images,val_labels)
