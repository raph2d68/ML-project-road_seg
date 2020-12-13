
from parameters import *
from helper import *
from model import model

import cv2

import imageio
import matplotlib.image as mpimg
import matplotlib as plt

import numpy as np
import tensorflow as tf
#import tensorflow_addons as tfa
from scipy import ndimage, misc


def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = load_image(image_filename)
            
            rot = tf.image.rot90(img)
            imageio.imwrite(image_filename + '../tests/' + 'test_img' + str(i).zfill(4) + '.png', rot)

            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return numpy.asarray(data)


# Extract label images
def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)

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


data_dir = 'ressource_files/training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/' 

# Extract it into numpy arrays.
train_data = extract_data(train_data_filename, TRAINING_SIZE)
#train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)

image_dir = data_dir + 'images/'
groundt_dir = data_dir + 'groundtruth/'

files = os.listdir(image_dir) #return list of files of a directory

images_list = [load_image(image_dir + files[i]) for i in range(TRAINING_SIZE)] # create list of the names of the images 
groundt_list = [load_image(groundt_dir + files[i]) for i in range(TRAINING_SIZE)]   # create list of the names of the groundtruth (images and groundtruth names are the same)

num_aug = TRAINING_SIZE + 1 

for i in range(TRAINING_SIZE) :

    # load images
    img = images_list[i]
    groundt = groundt_list[i]

    #rot90
    rot90_i = tf.image.rot90(img)
    rot90_gt = tf.image.rot90(groundt)
    plt.imsave(image_dir + 'satImage_' + str(num_aug).zfill(4) + '.png', rot90_i)
    plt.imsave(groundt_dir + 'satImage_' + str(num_aug).zfill(4) + '.png', rot90_gt, cmap='Greys_r')
    num_aug+=1

    #rot180
    rot180_i = tf.image.rot90(rot90_i)
    rot180_gt = tf.image.rot90(rot90_gt)
    plt.imsave(image_dir + 'satImage_' + str(num_aug).zfill(4) + '.png', rot180_i)
    plt.imsave(groundt_dir + 'satImage_' + str(num_aug).zfill(4) + '.png', rot180_gt, cmap='Greys_r')
    num_aug+=1

    #rot270
    rot270_i = tf.image.rot90(rot180_i)
    rot270_gt = tf.image.rot90(rot180_gt)
    plt.imsave(image_dir + 'satImage_' + str(num_aug).zfill(4) + '.png', rot270_i)
    plt.imsave(groundt_dir + 'satImage_' + str(num_aug).zfill(4) + '.png', rot270_gt, cmap='Greys_r')
    num_aug+=1

    #vertical flip
    plt.imsave(image_dir + 'satImage_' + str(num_aug).zfill(4) + '.png', tf.image.flip_left_right(img))
    plt.imsave(groundt_dir + 'satImage_' + str(num_aug).zfill(4) + '.png', tf.image.flip_left_right(groundt), cmap='Greys_r')
    num_aug+=1


    #horizontal flip
    plt.imsave(image_dir + 'satImage_' + str(num_aug).zfill(4) + '.png', tf.image.flip_up_down(img))
    plt.imsave(groundt_dir + 'satImage_' + str(num_aug).zfill(4) + '.png', tf.image.flip_up_down(groundt), cmap='Greys_r')
    num_aug+=1

    #45° flip
    plt.imsave(image_dir + 'satImage_' + str(num_aug).zfill(4) + '.png', tf.image.rot270(tf.image.flip_up_down(rot90_i)))
    plt.imsave(groundt_dir + 'satImage_' + str(num_aug).zfill(4) + '.png', tf.image.rot270(tf.image.flip_up_down(rot90_gt)), cmap='Greys_r')

    #270° flip
    plt.imsave(image_dir + 'satImage_' + str(num_aug).zfill(4) + '.png', tf.image.rot270(tf.image.flip_up_down(rot270_i)))
    plt.imsave(groundt_dir + 'satImage_' + str(num_aug).zfill(4) + '.png', tf.image.rot270(tf.image.flip_up_down(rot270_gt)), cmap='Greys_r')

    #add noise 

    #Centering


