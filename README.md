# Flow2ML

[![Issues](https://img.shields.io/github/issues/flow2ml/Flow2ML)](https://github.com/flow2ml/Flow2ML/issues)
[![PRs](https://img.shields.io/github/issues-pr-raw/flow2ml/Flow2ML)](https://github.com/flow2ml/Flow2ML/pulls)
[![Forks](https://img.shields.io/github/forks/flow2ml/Flow2ML?)](https://github.com/flow2ml/Flow2ML/network/members) 
[![Stars](https://img.shields.io/github/stars/flow2ml/Flow2ML?)](https://github.com/flow2ml/Flow2ML/stargazers)
[![Contributors](https://img.shields.io/github/contributors/flow2ml/Flow2ML)](https://github.com/flow2ml/Flow2ML/graphs/contributors)
![PyPi Version](https://img.shields.io/pypi/v/pypi-download-stats.svg)
--- 

## Table Of Contents
- [Introduction](#Introduction)
- [Why Flow2Ml](#Why-Flow2Ml)
- [Programming-languages-and technologies used](#Programming-languages-and-technologies-used)  
- [Dependencies](#dependencies)
- [Open Source Programs that Flow2ML is a part of](#Open-Source-Programs)
- [Installation](#Installation)
- [Sample Code](#Sample-Code)
- [Contributing](#Sample-Code)
- [Contributors](#Contributors)

## Introduction 
<p>Write only a Few Lines of
Machine learning code using
Flow2Ml</p>
 
<p>Quickly design and customize pre-processing workflow in machine learning.
Obtain training, validating samples with only 3 lines of code using Flow2ML toolkit

Check Installation and sample code to flow into your ML model much faster and efficiently.</p>
 
## Why Flow2ML
<p>Flow2ML is an open-source library to make the machine learning process much simpler. It loads the image data and applies the given filters and returns train data, train labels, validation data, and validation labels.
For all these steps it just takes 3 lines of code. It mostly helps beginners in the field of machine learning and deep learning where the user would deal with image-related data.</p>


## Programming languages and technologies used:
1. Python
2. HTML
3. Numpy library
4. OpenCV
5. Machine Learning

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
    
## Open Source programs that Flow2ML is a part of:

<p align="center">
 <a>
 <img  width="50%" height="20%" src="https://raw.githubusercontent.com/GirlScriptSummerOfCode/MentorshipProgram/master/GSsoc%20Type%20Logo%20Black.png">


    
## Download all Dependencies by :
```pip install -r requirements.txt```

## Installation
Install Flow2ML python files via pip.

```sh
    $ pip install flow2ml==1.0.3
```


## Sample Code
```py
    # To be given input by the user.
    img_dimensions = (150,150,3)
    test_val_split = 0.1

    # Import flow2ml package
    from flow2ml import Flow

    # Give the Dataset and Data directories
    flow = Flow( 'dataset_dir' , 'data_dir' )

    # Define The Filers to be used
    filters = ["median", "laplacian", "gaussian", "sobelx", "sobely"]

    # Apply The Filters
    flow.applyFilters( filters )

    # Define The augmentation operations to be used
    operations = {'flip': 'horizontal', 'rotate': 90}

    # Apply The Augmentation
    flow.applyAugmentation( operations )

    # Obtain Train, Validation data splits
    (train_x, train_y, val_x, val_y) = flow.getDataset( img_dimensions, test_val_split )
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


## Contributing

If you want to contribute to Flow2Ml, Please look into issues and propose your solutions to them.
We promote contributions from all developers regardless of them being a beginner or a pro. 
We go by the moto 
<code><strong>Caffeinate☕|| Collaborate🤝🏼|| Celebrate🎊</strong></code>
before that, please read <a href="https://github.com/flow2ml/Flow2ML/blob/main/CONTRIBUTING.md">contributing guidelines</a>

## Contributors👩🏽‍💻👨‍💻

### Credits goes to these wonderful people:✨

<table>
	<tr>
		<td>
   <a href="https://github.com/flow2ml/Flow2ML/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=flow2ml/Flow2ML" />
</a>
		</td>
	</tr>
</table>
