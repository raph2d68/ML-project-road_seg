***
# Road Segmentation Project

This repository is a Machine Learning project that aims to detect roads from a set of satellite images. It implements a U-Net algoritm that classifies the input of RGB images set, assigning either the label "road" or "background" to each pixel. A training set of 100 satellite road images (400x400 pixels), with their corresponding groundtruths was used. As a test set we used 50 unlabeled satellite road images (608x608 pixels).

***
# Files Details

## Images : 

- **Ressource_folder** contains the test and train images, with their groundtruth.

## Scripts  : 

- **helper.py** contains additional functions that were used during the whole procedure.

- **parameters.py** provides the parameter set for the task.

- **pre_processing.py** contains images augmentation functions.

- **post_processing.py** contains functions to improve pixel classification with Hough Transform, kernel tuning and morphological operators.

- **unet.py** contains the unet architecture related functions.

- **ressource_files** folder contains training images, groudtruth images, testing images as well as provided helper functions to generate a submission for the challenge with 16x16 pixel patches.

- **predictions** folder contains the predicted binary images, and **visual** folder contains the overlay of the training images and pedictions.

## Requirements

All scripts have been coded with the following versions :

- Python 3.7
- Tensorflow 2.4
- Numpy 1.19
- OpenCV 4.4
- Matplotlib 3.3

>*Note:* The training procedure was conducted using Google Colab to access better computing resources, in particular the **12GB NVIDIA Tesla K80 GPU.**

## Run instructions :

Scripts to run :

- **training.py** contains the code used to train the U-Net model.
- **prediction.py** contains the code that generates a prediction from test images and a trained model. Toggle the parameter ```bool POSTPROCESS``` to enable or not the post-processing on the predicted images.

1. Modify your parameters, such as the paths or the epoch numbers in ```parameters.py```
1. Make sure you have the right packages, have a look at ```requirements.txt``` if necessary.
1. To save time during parameters tuning, the training can be done without prediction (testing) and submission. The following commands runs the training of the model, and then runs a prediction from the trained model.

First clone the repo and go to your directory.

```
$ git clone <url> 
$ cd dir
```

**1. To run training and prediction separately**

Train the model.
```
$ python training.py
```
Run the prediction, and run post-processing if enabled.
```
$ python prediction.py 
```

**2. To run training and prediction, and make a submission from test images**

Run the complete project
```
$ python run.py
```

## Results file

- **final_submission.csv** contains our final submission for the AIcrowd challenge.

***

# Authors

- Raphaël Ausilio
- Valentin Bigot
- Valentin Karam
