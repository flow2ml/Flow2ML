import os
import sys ,time
from .Data_Loader import Data_Loader
from .Filters import Filters
from .Data_Augumentation import Data_Augumentation

class Flow(Data_Loader,Filters,Data_Augumentation):
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
    # checks to see if the list contains a dictionary with apply_on_augmentation key
    apply_on_augmentation = [i for i in filters if isinstance(i, dict) == True]
    if apply_on_augmentation:
      apply_on_augmentation = apply_on_augmentation[0].get('apply_on_augmentation', False)
      print(apply_on_augmentation)
    
    progress = [(100/len(filters))*i for i in range(0,len(filters)) ]
    progress_i = 0

    for folder in self.classes:
      for filter in filters:
        paths = [os.path.join(self.dataset_dir ,self.data_dir, folder)]
        # if apply_on_augmentation is true, creates a list of folders of augmented images to apply the filters to
        if apply_on_augmentation:
          for operation in apply_on_augmentation:
            paths = [os.path.join(self.dataset_dir ,self.data_dir, folder, operation.title() + "Images")]
        
        # applies filters to all folders in paths (single folder if no apply_on_augmentation)
        for path in paths:
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
          
          elif filter == "bilateral":
            self.applybilateral(path)
      
        time.sleep(0.1)
        self.update_progress( progress[progress_i]/100.0, f"Filtered all images in {folder}" )
        progress_i += 1

      status1 = f"Applied all filters to {folder} ...\r\n"
      self.update_progress( 100/100.0, f"Filtered all images in {folder}"  )
      progress_i = 0
    print()

  def applyAugmentation(self,operations):
    
    ''' 
      Applies given data augmentation operations 
      Args : 
        operations : (dictonary) python dictionary containing key value pairs of operations and values to be applied to the image data.
    '''
    
    self.operations = operations
    # checks to see if apply_on_filters key if present
    apply_on_filters = self.operations.get('apply_on_filters', False)

    progress = [(100/len(operations))*i for i in range(0,len(operations)) ]
    progress_i = 0

    for folder in self.classes:
      for operation in self.operations:
        paths = [os.path.join(self.dataset_dir ,self.data_dir, folder)]
        # if apply_on_filters is true, creates a list of folders of filtered images to apply the augmentations to
        if apply_on_filters:
          for filter in apply_on_filters:
            paths = [os.path.join(self.dataset_dir ,self.data_dir, folder, filter.title() + "Images")]
        
         # applies augmentations to all folders in paths (single folder if no apply_on_filters)
        for path in paths:
          if operation == "flipped":
            self.applyFlip(path)

          if operation == "rotated":
            self.applyRotate(path)
          
          if operation == "sheared":
            self.applyShear(path)

          if operation == "inverted":
            self.applyInvert(path)

          if operation == "histogramequalised":
            self.applyHistogramEqualization(path)
            
          if operation == "CLAHE":
            self.applyCLAHE(path)
          
          if operation == "cropped":
            self.applyCrop(path)

          if operation == "scaled":
            self.applyScale(path)

          if operation == "zoomed":
            self.applyZoom(path)

          if operation == "greyscale":
            self.applyGreyscale(path)

          if operation == "eroded":
            self.applyErode(path)

          if operation == "dilated":
            self.applyDilate(path)

          if operation == "opened":
            self.applyOpen(path)

          if operation == "closed":
            self.applyClose(path)

          if operation == "thresholded":
            self.applyThreshold(path)

          if operation == "colorspace":
            self.changeColorSpace(path)

        time.sleep(0.1)
        self.update_progress( progress[progress_i]/100.0, f"Augmented all images in {folder}" )
        progress_i += 1

      status1 = f"Applied all operations to {folder} ...\r\n"
      self.update_progress( 100/100.0, f"Augmented all images in {folder}"  )
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

    # creates processedData folder in self.dataset_dir
    final_data_folder = os.path.join(self.dataset_dir,'processedData')
    try:
      os.mkdir(final_data_folder)    
      print("Creating processedData Folder")
    except Exception as e:
      print(f"Unable to create processedData folder due to {e}")

    # moves all the images to that folder
    for folder in self.classes:
      path = os.path.join(self.dataset_dir ,self.data_dir, folder)
      self.img_label = self.create_dataset(path)
    
    self.update_progress( 50/100.0, "Created Datasets" )

    # Prepare Numpy dataset
    self.dataset = self.prepare_dataset(img_dimensions,train_val_split,self.img_label)

    self.update_progress( 100/100.0,"Created Datasets" ) 

    return self.dataset
