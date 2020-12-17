# Machine Learning - EPFL - Road Segmentation

This project task consisted in the implementation of a binary classifier to correclty assign either 
`road=1` or `background=0` label to each pixel of Google Earth aerial images. The training set consisted in 100 images
with their respective groundtruth images where road are correclty established from background. 
To tackle this challenge,
1. We augmented the training dataset through a pre-processing procedure.
2. We trained a [U-net architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) for the segmentation procedure.
3. We implemented a post-processing algorithm to correct easily found miss-classified pixels.

To evaluate model performance, [F1 score](https://en.wikipedia.org/wiki/F1_score) was monitored by computing both Precision and Recall metrics.

Details are provided in the ML-RST-report.pdf 

The dataset is available at the [CrowdAI page](https://www.crowdai.org/challenges/epfl-ml-road-segmentation).

# Training 

The training procedure was performed with the use of Google Colab in order to access better computational ressources.

# Files Details

- **final_submission.csv** contains our final submission for the AIcrowd challenge.
- **helper.py** contains additional functions that were used during the whole procedure.
- **parameters.py** provides the parameter set for the task.
- **pre_processing.py** contains images augmentation functions.
- **post_processing.py** contains functions to improve pixel classification with Hough Transform, kernel tuning and morphological operators.
- **prediction.py** contains the code to generate a prediction from a test set of images and a trained model.
- **training.py** contains the code implemented to train the Unet model.
- **unet.py** contains the unet architecture related functions.

- **ressource_files** folder contains training images, groudtruth images, testing images as well as provided helper functions to generate a submission for the challenge with 16x16 pixel patches.

# Authors 

- Raphaël Ausilio
- Valentin Bigot
- Valentin Karam
