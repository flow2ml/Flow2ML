"""
# Horse or Human Image Classification using Keras and CNN.
This is a dataset from:

->     https://www.kaggle.com/sanikamal/horses-or-humans-dataset

For Training : Here we have 500 Horse images and 527 Human (Including Male and Female) images.
For validation : we have 128 Horse images and 128 Human images.

Problem Statement : Classify given image as Horse or Human.
Solution : To solve this problem we are going to use Deep Learning Algorithm that is CNN.

"""

# Import Libraries
import keras
from keras.preprocessing.image import ImageDataGenerator # for data augmentation

import matplotlib.pyplot as plt
import os

"""
Keras Preprocessing is the data preprocessing and data augmentation module of the Keras deep learning library. It provides utilities for working with image data, text data, and sequence data.

Image data augmentation is a technique that can be used to artificially expand the size of a training dataset by creating modified versions of images in the dataset.

The Keras deep learning neural network library provides the capability to fit models using image data augmentation via the ImageDataGenerator class.

Refer site for more details : https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/


"""

# Load Data


# printing name of all available directories
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)

for dirname, _, filenames in os.walk('/kaggle/input'):
    Image_Count = 0

    for file in filenames:
        Image_Count += 1

    if Image_Count > 0:
        print('Total Files in directory {} is {}'.format(dirname, Image_Count))

#Storing the paths of test and train data as 2 variables
train_path = '/kaggle/input/horses-or-humans-dataset/horse-or-human/train'
val_path = '/kaggle/input/horses-or-humans-dataset/horse-or-human/validation'




"""
Data Processing

Here we defined our Image Data Generator function with some parameters.

* The 'directory' must be set to the path where your ‘n’ classes of folders are present.
* The 'target_size' is the size of your input images, every image will be resized to this size.
* 'color_mode': if the image is either black and white or grayscale set “grayscale” or if the image has three color channels, set “rgb”.
* 'batch_size': No. of images to be yielded from the generator per batch.
* 'class_mode': Set “binary” if you have only two classes to predict, if not set to“categorical”, in case if you’re developing an Autoencoder system, both input and the output would probably be the same image, for this case set to “input”.
* 'shuffle': Set True if you want to shuffle the order of the image that is being yielded, else set False.
* 'seed': Random seed for applying random image augmentation and shuffling the order of the image.
"""

training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'

)

"""
Now its time to generate the training images.. so will pass the directory path itself to our function, and in return we will get the images with different angle of single image.
Note : Source images are of diffeent size.. where we we want the result or output to be in same size for that will provide target_size.
flow_from_directory() expects at least one directory under the given directory path.
"""

training_data = training_datagen.flow_from_directory(
    directory = train_path,
    target_size = (150, 150),   # Setting the size of output images to have in same size.
    batch_size = 32,
    class_mode = 'binary',
    shuffle=True,
    seed=42 # set seed=42 to get the same results as this code
)


"""
So now the training_data is generated...
We can check the count of images found at given path.. as we already defined.. we have 500 images of Horse + 527 images of Human, which is 1027 total images.
The good thing is Keras itself set the class index for us, and this we can validate using 'class_indices'.
"""
# you can run this on a notebook after un-commenting
# training_data.class_indices
#
# training_data

"""
For Validation Dataset...
* 'class_mode': Set this to None, to return only the images.
* 'shuffle': Set this to False, because you need to yield the images in “order”, to predict the outputs and match them with their unique ids or filenames.
"""

# Doing the same for validation dataset.
# But here we do not need to generate the images for Validation, so we just use rescale.
valid_datagen = ImageDataGenerator(rescale = 1./255)
valid_data = valid_datagen.flow_from_directory(
    directory = val_path,
    target_size = (150, 150),   # Setting the size of output images to have in same size.
    batch_size = 32,
    class_mode = 'binary'
)

"""
# Plot the Images
"""

def plotImages(images_arr):
    fig, axes = plt.subplots(1,5,figsize = (20, 20))
    axes = axes.flatten()

    for img, ax in zip(images_arr, axes):
        ax.imshow(img)

    plt.tight_layout()  # to properly and cleanly show the images
    plt.show()          # shows the images

images_to_show = [training_data[0][0][0] for i in range(5)] # taking 5 images.
plotImages(images_to_show)

"""
Here we see the same image but with different variations in it.
"""

"""
Buidling the CNN Model
"""

cnn_model = keras.models.Sequential(
    [
        keras.layers.Conv2D(filters = 32, kernel_size = 3, input_shape = [150, 150, 3]),
        keras.layers.MaxPooling2D(pool_size = (2,2)),
        keras.layers.Conv2D(filters = 64, kernel_size = 3),
        keras.layers.MaxPooling2D(pool_size = (2,2)),
        keras.layers.Conv2D(filters = 128, kernel_size = 3),
        keras.layers.MaxPooling2D(pool_size = (2,2)),
        keras.layers.Conv2D(filters = 256, kernel_size = 3),
        keras.layers.MaxPooling2D(pool_size = (2,2)),

        keras.layers.Dropout(0.5),

        # Neural Network Building
        keras.layers.Flatten(),
        keras.layers.Dense(units = 128, activation = 'relu'), # Input Layer
        keras.layers.Dropout(0.1),
        keras.layers.Dense(units = 256, activation = 'relu'), # Hidden Layer
        keras.layers.Dropout(0.25),
        keras.layers.Dense(units = 2, activation = 'softmax'), # Output Layer
    ]
)


"""
# Compile Model
"""


from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


cnn_model.compile(
    optimizer = Adam(lr=0.0001),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)


# model_path = '/kaggle/input/output/horse_human_model.h5'
model_path = '/kaggle/working/horse_human_model.h5'
# model_path = '/kaggle/input/horse_human_model.h5'
checkpoint = ModelCheckpoint(model_path, monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = 'max')
callbacks_list = [checkpoint]


history = cnn_model.fit(
    training_data,
    epochs = 100,
    verbose = 1,
    validation_data = valid_data,
    callbacks = callbacks_list
)


"""
Visualization

We have captured all details like accuracy of train and validation dataset.
Similarly we do have Loss for both training and validation dataset.
So lets visualize it by plotting it.
"""

# Summarize the accuracy.

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch or iteration')
plt.legend(["train", 'valid'], loc = 'upper left')
plt.show()

# Summarize the Loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Accuracy')
plt.xlabel('epoch or iteration')
plt.legend(["train", 'valid'], loc = 'upper left')
plt.show()
