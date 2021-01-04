import os

from .Data_Loader import Data_Loader
from .Filters import Filters

class Flow(Data_Loader,Filters):
  '''
    This class connects the work flow. It calls various methods from 
    the inherited classes whenever required and completes the workflow.

    Inherits:
      Data_Loader: (Class) Contains various methods to handle data.
      Filters: (Class) Contains methods to apply filters to the given data.
  '''

  def __init__(self,dataset_dir,data_dir):
    '''
      Initializes the Flow class with all the required information that's 
      to be sent to the methods when they are called.
      Args :
        dataset_dir : (string) the datset containing Flow folder.
        data_dir : (string) the data containing folder with various labels
                            as sub folders.
    '''
    self.dataset_dir = dataset_dir
    self.data_dir = data_dir
    super().__init__(dataset_dir,data_dir)
    self.classes = self.getClasses()
    self.num_classes =  len(self.classes)


  def applyFilters(self,filters):
    ''' 
      Applies given filters 
      Args : 
        filters : (List) python list containing various filters to 
                         be applied to the image data.
    '''
    self.filters = filters
  
    for folder in self.classes:
      for filter in filters:
        path = os.path.join(self.dataset_dir ,self.data_dir, folder)
        if filter == "median":
          print(f"Applying Median Filter for {folder}")
          self.applyMedian(path)
        elif filter == "laplacian":
          print(f"Applying Laplacian Filter for {folder}")
          self.applylaplacian(path)
        elif filter == "sobelx":
          print(f"Applying Sobel-x Filter for {folder}")
          self.applysobelx(path)
        elif filter == "sobely":
          print(f"Applying Sobel-y Filter for {folder}")
          self.applysobely(path)
        elif filter == "gaussian":
          print(f"Applying Gaussian Filter for {folder}")
          self.applygaussian(path)

  def getDataset(self,img_dimensions,train_val_split):
    '''
      Generates the dataset.
      Moves all the images to a seperate folder and prepares a numpy dataset.

      Args:
        img_dimensions: (tuple) holds dimensions of the image after resizing.
        train_val_split: (float) holds train validation split value.
      
      Returns:
        train_val_dataset: (tuple) contains the numpy ndarrays.
                            (trainData, trainLabels, valData, valLabels).
    '''

    # creates processedData folder in self.dataset_dir and
    # moves all the images to that folder
    for folder in self.classes:
      path = os.path.join(self.dataset_dir ,self.data_dir, folder)
      self.img_label = self.create_dataset(path)

    # Prepare Numpy dataset
    self.dataset = self.prepare_dataset(img_dimensions,train_val_split,self.img_label)

    return self.dataset