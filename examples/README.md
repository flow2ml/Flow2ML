# Sample Code

## Image Datasets:

### Filters and augmentation:
```py
# Import flow2ml package
from flow2ml import Flow

# Give the Dataset and Data directories
flow = Flow( 'dataset_dir' , 'data_dir' )

# Define The Filters to be used
filters = ["median", "laplacian", "gaussian", "sobelx", "sobely", "bilateral"]

# Apply The Filters
flow.applyFilters( filters )

# Define The augmentation operations to be used
   
operations = {'flipped': 'horizontal', 'rotated': 90, 'sheared': {'x_axis': 5, 'y_axis': 15}, 'cropped': [50, 100, 50, 100], 'scaled': 0.1, 'zoomed': 2, 'histogramequalised':True, 'greyscale': True, 'CLAHE':True, 'inverted':True, 'eroded':True, 'dilated':True, 'opened':True, 'closed':True,'thresholded':{'type':'adaptive','thresh_val':0}, 'colorspace':{'input':'BGR','output':'BGR'}, 'canny':{'threshold_1':100,'threshold_2':200}}

# Apply The Augmentation
flow.applyAugmentation( operations )

# To apply augmentations on filtered images, supply a list of filters to apply the augmentations to using the apply_on_filters key. The below code applies flipping and rotation augmentation on median filtered images.
filters = ["median", "laplacian"]
flow.applyFilters( filters )
operations = {'flipped': 'horizontal', 'rotated': 90, 'apply_on_filters': ['median']}
flow.applyAugmentation( operations )

# To apply filters on augmented images, supply a list of augmentations to apply the filters to using the apply_on_augmentation key. The below code applies median and laplacian filters on the flipped images.
operations = {'flipped': 'horizontal', 'rotated': 90}
flow.applyAugmentation( operations )
filters = ["median", "laplacian", {'apply_on_augmentation': ['flipped']}]
flow.applyFilters( filters )
```

### Train and test splits:
```py
# To be given input by the user.
img_dimensions = (150, 150, 3)

# If working with greyscale image, change the number of channels from 3 to 1.
test_val_split = 0.25

# Obtain Train, Validation data splits
(train_x, train_y, val_x, val_y) = flow.getDataset( img_dimensions, test_val_split )
```

### Assesing image quality:
```py
# Create an image quality report using Entropy or BRISQUE for all images
image_quality = "entropy"
flow.calculateImageQuality( image_quality )
```

### Evaluating models:
```py
# For Pytorch and scikit-learn models
from flow2ml import Auto_Results

# Set the Input Model by replacing None
model = None

x = Auto_Results(model, val_x, val_y)

# Call the get_results_docx() function to get the results in a Results folder 
x.get_results_docx()

# For Tensorflow models 
from flow2ml import Tf_Results

# Set the Input Model by replacing None
model = None
x = Tf_Results(model, validation_generator)
x.tf_get_results_docx() 
```

### Model conversions:
```py
# Define The conversions to be used
conversions = {'tfjs':True, 'tflite':True}

# convert tensorflow model to tfjs/tflite
flow.deployTensorflowModels( conversions, model )
```

Please try to maintain the dataset in the following manner in order to run the code easily.

```text
dataset_dir
├──data_dir/
|       ├──Label 1 Folder
|       ├──Label 2 Folder
|       ├──Label 3 Folder  
|               .
|               .
|               .        
|       └──Label n Folder 
| 
└────Other Files
```

## Non image datasets:

### Working with .csv files:
```py
# Import the required module
from flow2ml import Process_Csv
import pandas as pd

# Read a csv file
df = pd.read_csv('./Tips.csv')

# Pass the dataframe to the module to generate analysis reports
x = Process_Csv(df)
```