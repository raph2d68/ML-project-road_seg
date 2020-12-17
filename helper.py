import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
import code
import matplotlib.pyplot as plt
import tensorflow.python.platform
import numpy as np
import tensorflow as tf

from PIL import Image
from parameters import *


#return an image data array (M, N, 3)
def get_image(img):
    return mpimg.imread(img)

#Convert array data pixel, into value between 0-255 
def img_float_to_uint8(img):    
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg

# Return an image containing two images side by side
# Input:        img         Input real image
#               gt_img      Predicted groundtruth Image
# Output:       cimg        concatenation of to images    
def concatenate_images(img, gt_img):
    n_channels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if n_channels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

# Superimposes an input image and the label prediction for this image
# Input:        img                 Input real image
#               predicted_img       Predicted groundtruth Image
# Output:       new_img             Superimposition of the two images
def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

# Compute the F1 score from the epoch history of the model, for testing set
# Input:    epo_hist        contain Recall and Precision
# Output:   f1_s            F1 score  
def f1_scores(epo_hist):
    pre = np.array(epo_hist.history['precision'])
    rec = np.array(epo_hist.history['recall'])
    f1_s = 2 * pre * rec / (pre + rec)
    return f1_s


# plot the F1-score in function of epoch
def plot_metric_history(f1_scores):
    plt.plot(f1_scores)
    plt.xlabel('# epochs')
    plt.ylabel('F1-Score')
    plt.title('F1-Score for every epochs')
    plt.savefig('F1-Scores.png')
