import glob
import cv2
import imageio
import matplotlib.image as mpimg
import matplotlib as plt
import numpy as np
import tensorflow as tf

from scipy import ndimage, misc
from scipy.ndimage.interpolation import rotate

from parameters import *
from helper import *


# Return a list of images located in file "filename"
# Input:    filename  folder where image are located
#           names     list of the names of the files to be loaded
# Output:    imgs      list of images corresponding to names
def load_image(filename, names):
    imgs=[]
    for i, idx in enumerate(names) :
        imageid = names[i]
        image_filename = filename + imageid 
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')
    return imgs


# Return a list of names of all png image in a folder, that contain "satImage" in their name
# Input:    train_data_filename     folder where image are located
#           TRAINING_SIZE           number of images to select 
#           rand_bool               if rand_bool==1 images are randomly selected, 
#                                   else selection depend on their name                 
# Output:   names                   list of names of selected images 
def get_data_names( train_data_filename, TRAINING_SIZE, rand_bool ):
    names = [os.path.basename(x) for x in glob.glob( train_data_filename +'satImage*.png')]
    if rand_bool:
        names = np.random.permutation(names)[0:TRAINING_SIZE]
    else: 
        names = names[0:TRAINING_SIZE]
    return names


# Extend the input data by several images transformations
# Input :   train_data_filename,    Folder containing train data images
#           train_labels_filename,  Folder containing label data images
#           names                   List of names of input images 
# Output :  augmented_imgs,         Augmented input images  [TRAINING_SIZE*11, n, m, 3 ] 
#           augmented_gt_imgs       Augmented groundtruth images  [TRAINING_SIZE*11, n, m ]
def augment_data( train_data_filename, train_labels_filename, names ):

    training_imgs    =  load_image(train_data_filename, names)
    groundtruth_imgs =  load_image(train_labels_filename, names)
   
    augmented_imgs = training_imgs
    augmented_gt_imgs = groundtruth_imgs   

    #rot90
    rot90 = np.rot90( training_imgs, k=1, axes=(1, 2))
    rot90_gt = np.rot90( groundtruth_imgs, k=1, axes=(1, 2))
    
    #rot180
    rot180 = np.rot90( training_imgs, k=2, axes=(1, 2))
    rot180_gt = np.rot90( groundtruth_imgs, k=2, axes=(1, 2))
    
    #rot270
    rot270 = np.rot90( training_imgs, k=3, axes=(1, 2))
    rot270_gt = np.rot90( groundtruth_imgs, k=3, axes=(1, 2))

    #up_down flip
    flipud = np.flip( training_imgs,  1 )
    flipud_gt = np.flip( groundtruth_imgs, 1 )

    #horizontal flip
    fliplr =    np.flip( training_imgs,  2 )
    fliplr_gt = np.flip( groundtruth_imgs, 2 )
   
    #45° flip
    flip45    = flip_to_45( training_imgs )
    flip45_gt = flip_to_45( groundtruth_imgs )
    
    #135° flip
    flip135 =    flip_to_135( training_imgs )
    flip135_gt = flip_to_135( groundtruth_imgs )
    
    #gaussian noise
    gauss =     add_gaussian_noise( training_imgs )
    gauss_gt =  groundtruth_imgs

    #rotation 45°
    rot45    = imrot(training_imgs,45)
    rot45_gt = gtrot(groundtruth_imgs,45)

    #rotation -45°
    rot315    = imrot(training_imgs,-45)
    rot315_gt = gtrot(groundtruth_imgs,-45)

    ''' These 3 augmentations have been added after deadline'''
    #rotation 15° without resizing
    rot15    = rotate(training_imgs, axes = (1,2), angle=15, reshape=False)
    rot15_gt = rotate(groundtruth_imgs, axes = (1,2), angle=15, reshape=False)

    #rotation 30° without resizing
    rot30    = rotate(training_imgs, axes = (1,2), angle=30, reshape=False)
    rot30_gt = rotate(groundtruth_imgs, axes = (1,2), angle=30, reshape=False)
    
    #rotation 60° without resizing
    rot60    = rotate(training_imgs, axes = (1,2), angle=60, reshape=False)
    rot60_gt = rotate(groundtruth_imgs, axes = (1,2), angle=60, reshape=False)

    augmented_imgs = np.concatenate((augmented_imgs, rot90, rot180, rot270, flipud, fliplr, flip45, 
                            flip135, gauss, rot45, rot315, rot15, rot30, rot60 ), axis=0, out=None)
    augmented_gt_imgs = np.concatenate((augmented_gt_imgs, rot90_gt, rot180_gt, rot270_gt, flipud_gt, fliplr_gt,
             flip45_gt, flip135_gt, gauss_gt, rot45_gt, rot315_gt, rot15_gt, rot30_gt, rot60_gt), axis=0, out=None)

    return  augmented_imgs, augmented_gt_imgs 


# Rotate an input image array by "ang"" degree, then crop en resize
# Input:  images        input data images array  [TRAINING_SIZE, n, m, 3 ]   
#         ang           angle of rotation in degree
# Output: rot45_resize  output data images array  [TRAINING_SIZE, n, m, 3 ]  
def imrot(images,ang) : 
    rot_45 = rotate(images, axes = (1,2), angle=ang, reshape=True) # rotate + or -45

    down_limit = 140    # down limit to crop is a quarter of the image = 400*1.4/4
    up_limit = 420      # up limit is 3 quarters = 400*1.4*3/4
    crop_45 = rot_45[:,down_limit:up_limit,down_limit:up_limit,:] # crop the image

    zoom_factor = 400/crop_45.shape[1]
    rot45_resize = ndimage.zoom(crop_45, zoom=(1,zoom_factor,zoom_factor,1))

    return rot45_resize

# Rotate an input groundtruth array by "ang"" degree, then crop en resize
# Input:  images        input data groundtruth array  [TRAINING_SIZE, n, m ]   
#         ang           angle of rotation in degree
# Output: rot45_resize  output data groundtruth array  [TRAINING_SIZE, n, m ]
def gtrot(images,ang) : 
    rot_45 = rotate(images, axes = (1,2), angle=ang, reshape=True) # rotate + or -45

    down_limit = 140    # down limit to crop is a quarter of the image
    up_limit = 420      # up limit is 3 quarters
    crop_45 = rot_45[:,down_limit:up_limit,down_limit:up_limit] # crop the image

    zoom_factor = 400/crop_45.shape[1]
    rot_45_resize = ndimage.zoom(crop_45, zoom=(1,zoom_factor,zoom_factor))

    return rot_45_resize

# Add gaussian noise to an input data array
def add_gaussian_noise(images):
    noise = np.random.normal(0,0.3, np.shape(images))
    return np.clip(images + np.float32(noise), 0, 1)

# Flip diagonaly an input data array
def flip_to_45(images):
    fliplr = np.flip(images, 2)
    out = np.rot90(fliplr, k=1, axes=(1, 2)) 
    return out

# Flip diagonaly an input data array
def flip_to_135(image):
    flipud = np.flip(image, 1)
    out = np.rot90(flipud, k=1, axes=(1, 2)) 
    return out 