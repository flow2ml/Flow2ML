# To be given input by the user.
img_dimensions = (150,150)
test_val_split = 0.1

# Import flow2ml package
from flow2ml import Flow

# Give the Dataset and Data directories
flow = Flow( 'dataset_dir' , 'data_dir' )

# Define The Filers to be used
filters = ["median", "laplacian", "gaussian", "sobelx", "sobely"]

# Apply The Filters
flow.applyFilters( filters )

# Obtain Train, Validation data splits
(train_x, train_y, val_x, val_y) = flow.getDataset( img_dimensions, test_val_split )