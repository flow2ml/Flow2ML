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
    return training_classes


  def create_dataset(self,classPath):
    '''
      moves all the processed images to a single folder and 
      creates a dictonary with image and its ground truth value.

      Args :
        classPath : (string) directory containing images for a particular class.

      Returns :
        dictonary. A python dictonary containing imageName as key
                   and one hot encoded class name as value
    '''

    final_data_folder = os.path.join(self.dataset_dir,'processedData')
    

    try:
      os.mkdir(final_data_folder)    
      print("Creating processedData Folder")
    except:
      pass

    for image in list(os.listdir(classPath)):

        if (image.find("Images") == -1):
          pass
          # path is a image

          shutil.move(classPath+"/"+image,final_data_folder)

          folderName = list(classPath.split("/"))[2]


          if (folderName == 'Bacterial leaf blight'):
            self.img_label[image] = np.asarray([1,0,0], dtype=np.float32)
          elif (folderName == 'Brown spot'):
            self.img_label[image] = np.asarray([0,0,1], dtype=np.float32)
          else:
            self.img_label[image] = np.asarray([0,1,0], dtype=np.float32)


        else:
          # path is a folder

            for image_in_folder in (list(os.listdir(os.path.join(classPath,image)))):
              

              filtered_image = os.path.join(classPath,image,image_in_folder)
              shutil.move(filtered_image,final_data_folder)

              folderName = list(classPath.split("/"))[2]


              if (folderName == 'Bacterial leaf blight'):
                self.img_label[image_in_folder] = np.asarray([1,0,0], dtype=np.float32)
              elif (folderName == 'Brown spot'):
                self.img_label[image_in_folder] = np.asarray([0,0,1], dtype=np.float32)
              else:
                self.img_label[image_in_folder] = np.asarray([0,1,0], dtype=np.float32)

    return self.img_label

  def resize_image(self,img_resize_shape,image_path):
    '''
      method to resize the image to the given arguments.
      Args:
        img_resize_shape: (tulpe). The images are
                           resized to the given size.
        image_path : (string). image name
    '''

    image_path = os.path.join(self.final_data_folder,image_path)

    # Read Image
    img = plt.imread(image_path) 

    # Resize Image
    img = cv2.resize(img,img_resize_shape)

    # saving the image.
    cv2.imwrite(image_path, img)
    


  def prepare_dataset(self,img_resize_shape,train_val_split,img_label_dict):
    '''
      Resizes the dataset and Prepares train, validation sets.
      Args :
         img_resize_shape: (tulpe). The images are resized to the given size.

         train_val_split: (float). value used to split the entire dataset
                                   to train and validation sets.

         img_label_dict: (dictonary). contains image name as key and its 
                                      label as value  

    '''
    print("Creating train val splits")
    self.img_label = img_label_dict
    
    # differentiating the complete dataset into training and validating datasets.
    img_name_train, img_name_val, output_label_train, output_label_val = train_test_split(
                                                                    list(self.img_label.keys()),
                                                                    list(self.img_label.values()),
                                                                    test_size=train_val_split,
                                                                    random_state=0)
    


    # Creating Numpy Dataset
    ( Height, Width ) = img_resize_shape
    channels = 3
    train_images = np.ndarray(shape=(len(img_name_train), Height, Width, channels), dtype=np.float32)
    train_labels = np.ndarray(shape=(len(output_label_train), self.num_classes ), dtype=np.float32)
    val_images = np.ndarray(shape=(len(img_name_val), Height, Width, channels), dtype=np.float32)
    val_labels = np.ndarray(shape=(len(output_label_val), self.num_classes ), dtype=np.float32)

    print()
    print("Creating training dataset")

    # Adding Values to the numpy datasets
    i=0

    for image in list(img_name_train):
      self.resize_image(img_resize_shape,image)
      image_path = os.path.join(self.final_data_folder,image)
      # Read Image
      img = cv2.imread(image_path) 
      train_images[i] = img
      train_labels[i] = np.asarray(output_label_train[i])
      i += 1


    print()
    print("Creating validating dataset")

    i=0

    for image in list(img_name_val):
      self.resize_image(img_resize_shape,image)
      image_path = os.path.join(self.final_data_folder,image)
      # Read Image
      img = cv2.imread(image_path) 
      val_images[i] = img
      val_labels[i] = np.asarray(output_label_val[i])
      i += 1

    return (train_images,train_labels,val_images,val_labels)
