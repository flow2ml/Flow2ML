# Flow2ML

---

## What's included

Here you can see the files present in this repository and their uses

```text

Flow2Ml
└────flow2ml/
        ├──__init__.py
        ├──Data_Loader.py
        ├──Filters.py  
        ├──Data_Augumentation.py
        ├──Flow.py 
        ├──setup.cfg
        ├──README.md
        └──license.txt
```
<hr>
Data_Loader.py, Filters.py, Data_Augumentation.py contains respective classes which holds various methods to deal with image data. 

<h3>Data_Loader.py</h3>
It contains the following methods to handle data<br>
<ol>
    <li>getClasses<br>Get Class Names From the data folder.</li>
    <li>create_dataset<br>Creates a processedData folder in the root directory and moves all the processed image files into that.</li>
    <li>resize_image<br>Resizes the image to the given dimensions.</li>
    <li>prepare_dataset<br>Creates training and validating numpy datasets.</li>
</ol>

<h3>Filters.py</h3>
Applies the following filters to the images in data directory<br>
<ol>
    <li>median</li>
    <li>laplacian</li>
    <li>gaussian</li>
    <li>sobel-x</li>
    <li>sobel-y</li>
</ol>

<h3>Data_Augmentation.py</h3>
Applies the following augmentation operations to the images in data directory<br>
<ol>
    <li>flipping</li>
    <li>rotation</li>
</ol>
<br><hr><br>
Flow.py contains Flow class which connects various other classes and maintains the work flow.

<h3>Flow.py</h3>
It contains the following methods to connect various dots<br>
<ol>
    <li>applyFilters. <br> It takes a list of filters and applies all of them and stores inside the data directory.</li>
    <li>applyAugmentation. <br> It takes a dictionary of augmentation operations and applies all of them and stores inside the data directory.</li>
    <li>getDataset. <br> It moves all the processed images into a new folder located in the root folder and creates training and validating numpy datasets.</li>
</ol>
