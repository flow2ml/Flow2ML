import os
import sys ,time
import tensorflowjs as tfjs
import tensorflow as tf
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

  def applyAugmentation(self,operations):
    
    ''' 
      Applies given data augmentation operations 
      Args : 
        operations : (dictonary) python dictionary containing key value pairs of operations and values to be applied to the image data.
    '''
    
    self.operations = operations
    progress = [(100/len(operations))*i for i in range(0,len(operations)) ]
    progress_i = 0

    for folder in self.classes:
      for operation in self.operations:
        path = os.path.join(self.dataset_dir ,self.data_dir, folder)
        
        if operation == "flip":
          self.applyFlip(path)

        if operation == "rotate":
          self.applyRotate(path)
        
        if operation == "shear":
          self.applyShear(path)

        if operation == "invert":
          self.applyInvert(path)

        if operation == "Hist_Equal":
          self.applyHistogramEqualization(path)
          
        if operation == "CLAHE":
          self.applyCLAHE(path)
        
        if operation == "crop":
          self.applyCrop(path)

        if operation == "scale":
          self.applyScale(path)

        if operation == "zoom":
          self.applyZoom(path)

        if operation == "greyscale":
          self.applyGreyscale(path)

        if operation == "erode":
          self.applyErode(path)

        if operation == "dilate":
          self.applyDilate(path)

        if operation == "open":
          self.applyOpen(path)

        if operation == "close":
          self.applyClose(path)

        if operation == "threshold":
          self.applyThreshold(path)

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

  def deployTensorflowModels(self,conversions,model):
    '''Deploy conversion from tensorflow model to tensorflowjs or tflite model'''

    if(conversions['tfjs']==True):
      # Applying the conversion function to the input model and converted tfjs model will be stored in 'trained_models' folder.
      tfjs.converters.save_keras_model(model, 'trained_models') 

    elif(conversions['tflite']==True):
      TF_LITE_MODEL_FILE_NAME='tf_lite_model.tflite'
      # Defining the convertor
      tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
      # Applying the convert function
      tflite_model = tf_lite_converter.convert()
      trained_models=TF_LITE_MODEL_FILE_NAME
      open(trained_models,"wb").write(tflite_model)