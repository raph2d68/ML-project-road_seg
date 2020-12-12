
from parameters import *
from helper import *
from model import model

import matplotlib.pyplot as plt

#opening followed by a closing
#opening = erosion puis dilatation

import cv2

import numpy as np
import tensorflow as tf
from scipy import ndimage, misc



def morphological_op( binary_image ):
    cv2.imwrite('1 binary input.jpg', binary_image*255)
    #define kernel dimension
    width=IMG_PATCH_SIZE

    kernel_v = get_kernel(2*width, 0)
    kernel_v2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(width, 3*width))
    
    kernel_h = get_kernel(2*width, 90 )
    kernel_h2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3*width, width))

    #vertical ligne
    close_image_v  = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(width, width)))
    cv2.imwrite('2 vertical_closing.jpg',close_image_v*255)
    
    opened_image_v = cv2.morphologyEx(close_image_v, cv2.MORPH_OPEN, get_kernel(3*width, 0))
    cv2.imwrite('2 vertical_opening.jpg',opened_image_v*255)
    
    #horizontal_image
    close_image_h  = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(width, width)))
    cv2.imwrite('3 horizontal_closing.jpg',close_image_h*255)
    
    opened_image_h = cv2.morphologyEx(close_image_h, cv2.MORPH_OPEN, get_kernel(3*width, 90)) 
    cv2.imwrite('3 horizontal_opening.jpg', opened_image_h*255)

    sum_image = cv2.bitwise_or(opened_image_v, opened_image_h, mask = None) 
    sum_image = np.uint8((sum_image*255))
   
    #rho	Distance resolution of the accumulator in pixels.
    #theta	Angle resolution of the accumulator in radians.
    #threshold	Accumulator threshold parameter. Only those lines are returned that get enough votes ( >threshold ).
    rho = 1  
    theta = 3.14/360  
    threshold = 150
    cv2.imwrite('4 sum of opening and closing.jpg',sum_image)
   
    edges = cv2.Canny(sum_image, 5,100,apertureSize = 3)
    cv2.imwrite('5 edge_detection.jpg',edges)
    line = cv2.HoughLines(edges, 1, np.pi/360,150)
    line = np.squeeze(line)
    if np.size(line) > 1:
        #convert rad to degree
        b = line*360/(2*np.pi)
        if np.size(b)==2:
            b=np.array([b])
        angles_means = get_mean_angle_values(b[:,1])
        
        #proper closing
        #print("angle unique", angles_means)
        idx_of_max_angle=find_max_values(angles_means, 2)
        #print("idx_of_max_angle", idx_of_max_angle, 'type', type(idx_of_max_angle), "shape", np.shape(idx_of_max_angle) )
        reduced_angles_means = [ angles_means[i] for i in idx_of_max_angle]
        #print("reduced_unique_angle", reduced_angles_means)

        for i, angle in enumerate(reduced_angles_means):
            kernel_clean = get_kernel(3*width, angle)
            out  = cv2.morphologyEx(sum_image, cv2.MORPH_CLOSE, kernel_clean)
            #sum = cv2.bitwise_or(close_image_v, close_image_h, mask = None) 

        cv2.imwrite('7 output.jpg',out)
        return out/255
    else:
        print("no line hough detected")
        return sum_image/255  


def get_mean_angle_values( b ):

    tol=TOL_ANGLE  
    angles_mean=[]
    i=0
    full_len=len(b)
    while len(b)>0 :  
        smalltab=b[abs(b[0]-b)<tol]
        if(len(smalltab)>(full_len/25)):
            angles_mean.append(np.mean(smalltab))
        b = b[abs(b[0]-b)>=tol]
        i = i+1
    
    return angles_mean

def find_max_values(array, nb_of_max):
    
    index=[]
    np_array = np.array(array)
    new_np_array = np_array.copy()
    
    for i in range (nb_of_max):
        if len(new_np_array):
            max_value=np.max(new_np_array)
            max_idx=np.where(np_array==max_value)
            index.append(max_idx[0][0])
            mask=new_np_array!=np.max(new_np_array)
            new_np_array=new_np_array[mask]
  
    return np.array(index)



def get_kernel(width, angle):

    ker = np.ones((width, 10 ), np.uint8)
    rot_ker = ndimage.rotate(ker, 180-angle, reshape=True)
    #print(" kernel for angle", angle, rot_ker  )
  
    #cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(width+1, width+1))

    return rot_ker


def generate_output(s, filename, image_idx, all_nodes): 
    
    imageid = "satImage_%.3d" % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)
  
    print("IMG Idx", image_idx)

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