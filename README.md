***
# Road Segmentation Project

This repository is a learning machine project that aims to detect roads from a set of satellite images. It implements a U-Net that classifies a set of RGB images, assigning either the label "road" or "background" to each pixel. A training set of 100 satellite road images (400x400 pixels), with their corresponding groundtruths was used. As a test set we used 50 satellite road images (608x608 pixels).

***
# Files Details

## Images : 

- **Ressource_folder** contains the test and train images, with their groundtruth.

- **prediction** is a subfolder that contains the images used for the final prediction.

## Scripts  : 

- **helper.py** contains additional functions that were used during the whole procedure.

- **parameters.py** provides the parameter set for the task.

- **pre_processing.py** contains images augmentation functions.

- **post_processing.py** contains functions to improve pixel classification with Hough Transform, kernel tuning and morphological operators.

- **unet.py** contains the unet architecture related functions.

- **ressource_files** folder contains training images, groudtruth images, testing images as well as provided helper functions to generate a submission for the challenge with 16x16 pixel patches.

## Run instructions

All scripts have been coded with the following versions :

- Python 3.7
- Tensorflow 2.4
- Numpy 1.19
- OpenCV 4.4
- Matplotlib 3.3

Needed packages are available for download in the file ```requirements.txt```.

>*Note:* The training procedure was conducted using Google Colab to access better computing resources, in particular the **12GB NVIDIA Tesla K80 GPU.**

Scripts to run :

- **training.py** contains the code used to train the Unet model.
- **prediction.py** contains the code that generates a prediction from a test images set and a trained model. Toggle the parameter ```bool POSTPROCESS``` to enable or not the post-processing on the predicted images.

To save time during parameters tuning, the training can be done without submission. Example commands :

```
git clone <url> // clone the repo
cd dir
python training.py // train the model
python prediction.py // run the prediction, and 
```

## Results file

- **final_submission.csv** contains our final submission for the AIcrowd challenge.

***

# Authors

- Raphaël Ausilio
- Valentin Bigot
- Valentin Karam
