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
        ├──Auto_Results.py
        ├──Flow.py 
        ├──Image_Quality.py
        ├──setup.cfg
        ├──README.md
        └──license.txt
```
<hr>
Data_Loader.py, Filters.py, Data_Augumentation.py contains respective classes which holds various methods to deal with image data.<br> 
Auto_Results.py contains the class to deal with automated analysis of a trained model.

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
    <li>bilateral</li>
</ol>

<h3>Data_Augmentation.py</h3>
Applies the following augmentation operations to the images in data directory<br>
<ol>
    <li>flipping</li>
    <li>rotation</li>
    <li>shearing</li>
    <li>inverting</li>
    <li>HistogramEqualisation</li>
    <li>CLAHE<li>
    <li>cropping</li>
    <li>scaling</li>
    <li>zooming</li>
    <li>greyscale</li>
    <li>Erosion</li>
    <li>Dilation</li>
    <li>Opening</li>
    <li>Closing</li>
    <li>thresholding</li>
    <li>colorspace_Conversion</li>
</ol>
<br><hr><br>
Flow.py contains Flow class which connects various other classes and maintains the work flow.

<h3>Flow.py</h3>
It contains the following methods to connect various dots<br>
<ol>
    <li>applyFilters. <br> It takes a list of filters and applies all of them and stores inside the data directory.</li>
    <li>applyAugmentation. <br> It takes a dictionary of augmentation operations and applies all of them and stores inside the data directory.</li>
    <li>getDataset. <br> It moves all the processed images into a new folder located in the root folder and creates training and validating numpy datasets.</li>
    <li>deployTensorflowModels <br> It takes tensorflow model as input and converts into tensorflowjs or tensorflowlite model depending upon the user input.</li>
</ol>
<br><hr><br>

<h3>Auto_Results.py</h3>
It contains the following methods to get an automated analysis of a trained model<br>
<ol>
    <li>__init__ <br> It takes a trained model, test_x (test data) and test_y (test data labels) and creates a Results directory.</li>
    <li>roc_curve <br> It takes a filename and stores the roc curve plot inside the Results directory.</li>
    <li>confusion_matrix <br> Plots and saves the confusion matrix figure in the Results directory</li>
    <li>precision_recall_curve <br> It takes a filename and stores the precision recall curve plot inside the Results directory.</li>
    <li>get_results_docx <br> It calls all of the above functions with their deafult filenames and stores the figures along with a results.docx inside the Results directory.</li>
</ol>

<h3>Tf_Results.py</h3>
It contains the following methods to get an automated analysis of a trained model<br>
<ol>
    <li>__init__ <br> It takes a trained model, validation generator and creates a Results directory.</li>
    <li>roc_curve <br> It takes a filename and stores the roc curve plot inside the Results directory.</li>
    <li>confusion_matrix <br> Plots and saves the confusion matrix figure in the Results directory</li>
    <li>precision_recall_curve <br> It takes a filename and stores the precision recall curve plot inside the Results directory.</li>
    <li>get_results_docx <br> It calls all of the above functions with their deafult filenames and stores the figures along with a results.docx inside the Results directory.</li>
</ol>

<h3>Image_Quality.py</h3>
It contains the following methods to get an automated analysis of a trained model<br>
<ol>
    <li>__init__ <br> Initialises an empty dictionary to store images and their scores.</li>
    <li>generate_img_scores <br> It takes the path to the processedData folder and fills the dictionary with BRISQUE scores for all images in it.</li>
    <li>create_scores_doc <br> It saves the image quality report in the GeneratedReports directory.</li>
</ol>