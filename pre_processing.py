
from parameters import *
from helper import *
from model import model
import glob

import cv2

import imageio
import matplotlib.image as mpimg
import matplotlib as plt

import numpy as np
import tensorflow as tf
#import tensorflow_addons as tfa
from scipy import ndimage, misc


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


def get_data_names( train_data_filename, TRAINING_SIZE, rand_bool ):

    names = [os.path.basename(x) for x in glob.glob( train_data_filename +'satImage*.png')]
    
    if rand_bool:
        names = np.random.permutation(names)[0:TRAINING_SIZE]
    else: 
        names = names[0:TRAINING_SIZE]

    print(names)
    return names



def create_patches( data ):

    imgs=np.array(data)
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return np.asarray(data)


# Extract label images
def extract_labels( data ):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = np.array(data)

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)

#extraire des carres de w*h de l'image et les mettre les un a la suite des autres dans list_patches
# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches



'''
data_dir = 'ressource_files/training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/' 

# Extract it into np arrays.
train_data = extract_data(train_data_filename, TRAINING_SIZE)
#train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)

image_dir = data_dir + 'images/'
groundt_dir = data_dir + 'groundtruth/'

files = os.listdir(image_dir) #return list of files of a directory

''' 
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
   

    #vertical flip
    flipud = np.flip( training_imgs,  1 )
    flipud_gt = np.flip( groundtruth_imgs, 1 )
   

    #horizontal flip
    fliplr =    np.flip( training_imgs,  2 )
    fliplr_gt = np.flip( groundtruth_imgs, 2 )
   

    #45° flip
    flip45 =    flip_to_45( training_imgs )
    flip45_gt = flip_to_45( groundtruth_imgs )
  
    
    #135° flip
    flip135 =    flip_to_135( training_imgs )
    flip135_gt = flip_to_135( groundtruth_imgs )
  

    #gaussian noise
    gauss =     add_gaussian_noise( training_imgs )
    gauss_gt =  add_gaussian_noise( groundtruth_imgs )
  


    augmented_imgs = np.concatenate((rot90, rot180, rot270, flipud, fliplr, flip45, flip135, gauss), axis=0, out=None)
    augmented_gt_imgs = np.concatenate((rot90_gt, rot180_gt, rot270_gt, flipud_gt, fliplr_gt, flip45_gt, flip135_gt, gauss_gt), axis=0, out=None)


    return  augmented_imgs, augmented_gt_imgs 

def add_gaussian_noise(images):
    noise = np.random.normal(0, .1, np.shape(images))
    noise_img = images + np.int_(noise*255)
    #eventuellement ajouter un clip
    return noise_img

def flip_to_45(images):
    fliplr = np.flip(images, 2)
    out = np.rot90(fliplr, k=1, axes=(1, 2)) 
    return out

def flip_to_135(image):
    flipud = np.flip(image, 1)
    out = np.rot90(flipud, k=1, axes=(1, 2)) 
    return out 