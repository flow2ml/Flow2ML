# Flow2ML
--- 

## Table Of Contents
- [Introduction](#Introduction)
- [Why Flow2Ml](#Why-Flow2Ml)
- [Dependencies](#dependencies)
- [Installation](#Installation)
- [Sample Code](#Sample-Code)
- [Contributing](#Sample-Code)

## Introduction 
<p>Write only a Few Lines of
Machine learning code using
Flow2Ml</p>

<p>Quickly design and customize pre-processing workflow in machine learning.
Obtain training, validating samples with only 3 lines of code using Flow2ML toolkit
Check Installation and sample code to flow into your ml model fastly.</p>

## Why Flow2Ml
<p>Flow2ML is an open source library to make machine learning process much simpler. It loads the image data and applies the given filters and returns train data, train labels, validation data and validation labels.
For all these steps it just take 3 lines of code. It mostly helps beginners in the field of machine learning and deep learning where the user would deal with image related data.</p>

## Dependencies
Before Running the code you need to have certain packages to be installed. They are listed out here
    <ol>
        <li>cv2</li>
        <li>os</li>
        <li>shutil</li>
        <li>sklearn</li>
        <li>numpy</li>
        <li>matplotlib</li>
    </ol> 

## Installation
Install Flow2ML python files via pip.

```sh
    $ pip install flow2ml
```


## Sample Code
```py
    # To be given input by the user.
    img_dimensions = (150,150)
    test_val_split = 0.1

    # Import flow2ml package
    from flow2ml import Flow

    # Give the Dataset and Data directories
    flow = Flow( 'dataset_dir' , 'data_dir' )

    # Define The Filers to be used
    filters = ["median", "laplacian", "gaussian"]

    # Apply The Filters
    flow.applyFilters( filters )

    # Obtain Train, Validation data splits
    (train_x, train_y, val_x, val_y) = flow.getDataset( img_dimensions, test_val_split )
```

## Contributing
If you want to contribute to Flow2Ml, Please look into issues and propose your solutions to them.