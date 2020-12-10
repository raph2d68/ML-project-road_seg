
from parameters import *
from helper import *
from model import model

#opening followed by a closing
#opening = erosion puis dilatation

import cv2
import numpy as np
import tensorflow as tf





def morphological_op( binary_image ):

    #img = cv2.imread('j.png',0)

    #define kernel dimension
    width=IMG_PATCH_SIZE

    kernel_v = np.ones((width*5,width),np.uint8)
    kernel_v2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(width, 3*width))
    
    
    kernel_h = np.ones((width,width*5),np.uint8)
    kernel_h2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3*width, width))

    #opening followed by closing  
    #closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    #vertical ligne
    opened_image_v = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_v)
    close_image_v  = cv2.morphologyEx(opened_image_v, cv2.MORPH_CLOSE, kernel_v2)
   
    #horizontal_image

    opened_image_h = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_h) 
    close_image_h  = cv2.morphologyEx(opened_image_h, cv2.MORPH_CLOSE, kernel_h2)

    out = cv2.bitwise_or(close_image_v, close_image_h, mask = None) 

    return out



def kernel_shape(width):


    #return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(width+1, width+1))

    return np.ones((width+1,width),np.uint8)


def generate_output(s, filename, image_idx, all_nodes): 
    
    prediction_training_dir = "predictions_training/"
    imageid = "satImage_%.3d" % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)

    print("img_prediction IDX", image_idx)


    img_prediction = get_prediction(s, img, all_nodes)
    
    img_prediction_processed = morphological_op(img_prediction)
    
    #with post_processing

    pimg = get_prediction_with_groundtruth(img, img_prediction_processed, image_idx)
    Image.fromarray(pimg).save(prediction_training_dir + "prediction_pp_" + str(image_idx) + ".png")
    oimg = get_prediction_with_overlay(img, img_prediction_processed, image_idx)
    oimg.save(prediction_training_dir + "overlay_pp_" + str(image_idx) + ".png")   
    
    #without  post_processing

    pimg = get_prediction_with_groundtruth(img, img_prediction, image_idx)
    Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(image_idx) + ".png")
    oimg = get_prediction_with_overlay(img, img_prediction, image_idx)
    oimg.save(prediction_training_dir + "overlay_" + str(image_idx) + ".png")   

    return


        # Get prediction for given input image 
def get_prediction(s, img, all_nodes):

    
    data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
    data_node = tf.constant(data)
 
    output = tf.nn.softmax(model(data_node, all_nodes))
    output_prediction = s.run(output)
    img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

    return img_prediction


# Get a concatenation of the prediction and groundtruth for given input file
def get_prediction_with_groundtruth(img, img_prediction, image_idx):

   
    #print("img_prediction", img_prediction)
    cimg = concatenate_images(img, img_prediction)
    
    return cimg



# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay(img, img_prediction, idx):
    
    oimg = make_img_overlay(img, img_prediction)
    
    return oimg