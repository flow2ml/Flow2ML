import os
import sys ,time
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
      Initializes the main class with all the required information that's 
      to be sent to the methods when they are called.
      Args :
        dataset_dir : (string) the datset containing main folder.
        data_dir : (string) the data containing folder with various labels
                            as sub folders.
    '''
    self.dataset_dir = dataset_dir
    self.data_dir = data_dir
    super().__init__(dataset_dir,data_dir)
    self.classes = self.getClasses()

    try:
      self.classes.remove('.ipynb_checkpoints') 
    except:
      pass

    self.num_classes =  len(self.classes)


  def update_progress(self,progress,subStatus):
    '''
      Function used to update the progress bar in the console 
      Args :
        progress: float value for representing percentage
    '''
    barLength = 10 # Modify this to change the length of the progress bar
    status=""

    if progress >= 1:
        progress = 1
        status = subStatus +" ...\r\n"

    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
    

  def applyFilters(self,filters):
    ''' 
      Applies given filters 
      Args : 
        filters : (List) python list containing various filters to 
                         be applied to the image data.
    '''
    self.filters = filters
    progress = [(100/len(filters))*i for i in range(0,len(filters)) ]
    progress_i = 0

    for folder in self.classes:
      for filter in filters:
        path = os.path.join(self.dataset_dir ,self.data_dir, folder)
        
        if filter == "median":
          self.applyMedian(path)

        elif filter == "laplacian":
          self.applylaplacian(path)

        elif filter == "sobelx":
          self.applysobelx(path)

        elif filter == "sobely":
          self.applysobely(path)

        elif filter == "gaussian":
          self.applygaussian(path)
      
        time.sleep(0.1)
        self.update_progress( progress[progress_i]/100.0, f"Filtered all images in {folder}" )
        progress_i += 1

      status1 = f"Applied all filters to {folder} ...\r\n"
      self.update_progress( 100/100.0, f"Filtered all images in {folder}"  )
      progress_i = 0
    print()

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

    self.update_progress( 0/100.0, "Created Datasets" )  

    # creates processedData folder in self.dataset_dir and
    # moves all the images to that folder
    for folder in self.classes:
      path = os.path.join(self.dataset_dir ,self.data_dir, folder)
      self.img_label = self.create_dataset(path)
    
    self.update_progress( 50/100.0, "Created Datasets" )

    # Prepare Numpy dataset
    self.dataset = self.prepare_dataset(img_dimensions,train_val_split,self.img_label)

    self.update_progress( 100/100.0,"Created Datasets" ) 

    return self.dataset