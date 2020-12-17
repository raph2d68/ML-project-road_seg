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

