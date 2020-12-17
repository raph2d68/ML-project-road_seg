import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage, misc

from parameters import *
from helper import *
from pre_processing import *



# applies morphological operators on binary predicted image
# input:    binary image [n, m]
# output:   binary image [n, m]
def morphological_op( binary_image ):
    if POST_PROCESS:
        #define kernel dimension
        width=IMG_PATCH_SIZE

        rho = 1                     #  resolution of the parameter r in pixels 
        theta = np.pi/360          #  resolution of the parameter θ in radians
        threshold = 100             #  minimum number of intersections to detect a line
        minVal  = 5                 #  canny filter grad min and max treshold 
        maxVal  = 100
        #cv2.imwrite('4 sum of opening and closing.jpg',sum_image)

        #Applies edges detection and hough transform on input image
        binary_image = np.uint8((binary_image*255))
        edges = cv2.Canny(binary_image, minVal, maxVal, apertureSize = 3)
        
        line = cv2.HoughLines(edges, rho, theta, threshold)
        line = np.squeeze(line)
        
        if np.size(line) > 1:
            #convert rad to degree
            b = line*360/(2*np.pi)
            if np.size(b)==2:
                b=np.array([b])
            angles_means = get_mean_angle_values(b[:,1])
            idx_of_max_angle = find_max_values(angles_means, NUMBER_OF_KERNEL)
            reduced_angles_means = [ angles_means[i] for i in idx_of_max_angle]
        
            out_img = np.array(binary_image)/255
            for i, angle in enumerate(reduced_angles_means):
                #opening and then closing for all different angles
                out_op  = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, get_kernel(3*width, 2, angle))
                out_cl  = cv2.morphologyEx(out_op, cv2.MORPH_CLOSE, get_kernel(4*width, 2, angle))
                #summing all to input
                out_img = cv2.bitwise_or(out_img, out_cl/255, mask = None) 
            #round kernel to close   
            round_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10, 10))
            out_img  = cv2.morphologyEx(out_img, cv2.MORPH_CLOSE, round_ker)
            return out_img/255
        else:
            print("no line hough detected")
            return binary_image/255  
        
    return binary_image

# Average similar angles (of Hough Transform here) that are in a range tol  
# intput: angle_list   1-D list of angle 
# output: angles_mean  1-D list of mean of input angles
def get_mean_angle_values( angle_list ):
    tol=TOL_ANGLE  
    angle_mean=[]
    i=0
    full_len=len(angle_list)
    while len(angle_list)>0 :  
        smalltab=angle_list[abs(angle_list[0]-angle_list)<tol]
        if(len(smalltab)>(full_len/25)):
            angle_mean.append(np.mean(smalltab))
        angle_list = angle_list[abs(angle_list[0]-angle_list)>=tol]
        i = i+1
    return angle_mean


# Find index of "nb_of_max" maximum value in a 1-D array
# Intput:   array       1-D array
#           nb_of_max   Number of maximum value to extract from array
# Output:   index       1-D array of size nb_of_max   
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


# Create a rectangular kernel for morphological operation, tilted of an certain angle
# Intput:   width       Dimension of the kernel in pixel
#           angle       Angle of tilting of kernel in degree
# Output:   rot_ker     A titled kernel width*10 tilted by an angle "angle"
def get_kernel(height , width,  angle):
    ker = np.ones((height, width ), np.uint8)
    rot_ker = ndimage.rotate(ker, 180-angle, reshape=True)
    return rot_ker


# Generate output image with overlaying and concatenation of prediction
# Intput:   image_idx       index of the current image
#           img             current image                                   [n, m, 3 ]
#           img_prediction  binary model prediction for current image       [n, m ]     
def generate_output(image_idx, img, img_prediction ): 
    
    print("IMG Idx prediction", image_idx)
    dir_supp_images = PATH_PREDICTION_DIR + "visual/"
    os.makedirs(dir_supp_images , exist_ok=True)
    os.makedirs(PATH_PREDICTION_DIR, exist_ok=True)
    
    img_prediction_postprocessed = morphological_op(img_prediction)
    
    #with post_processing
    if POST_PROCESS:
        Image.fromarray(img_float_to_uint8(img_prediction_postprocessed)).save(
                                            PATH_PREDICTION_DIR + "prediction_pp_" + str(image_idx+1) + ".png")
        pimg = concatenate_images(img, img_float_to_uint8(img_prediction_postprocessed))
        Image.fromarray(pimg).save(dir_supp_images + "prediction_pp_" + str(image_idx+1) + ".png")
        oimg = make_img_overlay(img, img_prediction_postprocessed)
        #oimg.save(dir_supp_images + "overlay_pp_" + str(image_idx+1) + ".png")   
    else : 
        #without post_processing
        Image.fromarray(img_float_to_uint8(img_prediction)).save(
                                                PATH_PREDICTION_DIR + "prediction_" + str(image_idx+1) + ".png")
        pimg = concatenate_images(img, img_prediction)
        Image.fromarray(pimg).save(dir_supp_images + "prediction_" + str(image_idx+1) + ".png")
        
        oimg = make_img_overlay(img, img_prediction)
        #oimg.save(dir_supp_images + "overlay_" + str(image_idx+1) + ".png")   

    return
