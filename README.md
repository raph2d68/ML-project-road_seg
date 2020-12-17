***
# Road Segmentation Project

This repository is a Machine Learning project that aims to detect roads from a set of satellite images. It implements a U-Net algoritm that classifies the input of RGB images set, assigning either the label "road" or "background" to each pixel. A training set of 100 satellite road images (400x400 pixels), with their corresponding groundtruths was used. As a test set we used 50 unlabeled satellite road images (608x608 pixels).

***
# Files Details

## Images : 

- **ressource_files** folder contains needed images 
    - **/training** contains 100 400x400 RGB training images and groudtruths
    -  **/test_set_images** contains 50 608x608 RGB test images

## Scripts  : 

- **helper.py** contains additional functions that were used during the whole procedure.

- **parameters.py** provides the parameter set for the task.

- **pre_processing.py** contains images augmentation functions.

- **post_processing.py** contains functions to improve pixel classification with Hough Transform, kernel tuning and morphological operators.

- **unet.py** contains the unet architecture related functions.

- **prediction.py** contains the necessary function to predict labels from trained U-net model.

## Requirements

All scripts have been coded with the following versions :

- Python 3.7
- Tensorflow 2.4
- Numpy 1.19
- OpenCV 4.4
- Matplotlib 3.3

>*Note:* The training procedure was conducted using Google Colab to access better computing resources, in particular the **12GB NVIDIA Tesla K80 GPU.**

## Run instructions :

Scripts :

- **training.py** contains the code used to train the U-Net model.

- **run.py** makes a prediction with the test set and makes a submission. 
    - **prediction.py** contains the code that generates a prediction from test images and a trained model. Toggle the parameter ```bool POST_PROCESS``` to enable or not the post-processing on the predicted images.

Steps to run project :

1. Modify your parameters, such as the paths or the epoch numbers in ```parameters.py```
1. Make sure you have the right packages, have a look at ```requirements.txt``` if necessary.
1. To save time during parameters tuning, the training can be done without prediction (testing) and submission : 


First clone the repo and go to your directory.

```
$ git clone <url> 
$ cd dir
```
Then train the model.
```
$ python training.py
```

Now you can either make a submission, or simply make a prediction

**1. Run prediction separately from submission**


Run the prediction, and run post-processing if enabled in order to update ressource_files/prediction images.
```
$ python prediction.py 
```

**2. Run prediction, and make a submission from test images**

In order to obtain the csv file of our final_submission, use **run.py** without post_processing.
That will gather images from the ressource_files/prediction folder and generate a submission file.
```
$ python run.py
```

## Results file

- **final_submission.csv** contains our final submission for the AIcrowd challenge.
- **ressources_files/prediction** contains the predicted binary images as stated above 

***

# Authors

- Raphaël Ausilio
- Valentin Bigot
- Valentin Karam


# Reference

[Unet architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

### GitHub repositories

[EPFML_Project2](https://github.com/ntalabot/EPFML_Project2) 

[EPFL-Machine-Learning-Road-Segmentation](https://https://github.com/zghonda/EPFL-Machine-Learning-Road-Segmentation.google.com) 


