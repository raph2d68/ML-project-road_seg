
from parameters import *
from helper import *
from model import model

#opening followed by a closing
#opening = erosion puis dilatation

import cv2

import numpy as np
import tensorflow as tf
from scipy import ndimage, misc





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

    rho= 5
    theta= 3.14/180
    threshold=200
    line = hough_op( out, rho, theta, threshold)

    line=np.squeeze(line)
    print("line", np.shape(line))
        
    tol=30

    b=line[:,1]*360/(2*np.pi)
    print(b)
    b=b[b!=0]
    b[b>90]=b[b>90]-90

    
    unique_angles=b[~(np.triu(np.abs(b[:,None] - b) <= tol,1)).any(0)]
    
    
    print('b',unique_angles)
    print(np.shape(unique_angles))

    #proper closing
    if len(unique_angles)==2:
        kernel_h_clean = get_kernel(2*width, unique_angles[0])
        kernel_v_clean = get_kernel(2*width, unique_angles[1])
        
        print(kernel_h_clean)
        print(kernel_v_clean)
        close_image_h  = cv2.morphologyEx(close_image_h, cv2.MORPH_CLOSE, kernel_h_clean)
        close_image_v  = cv2.morphologyEx(close_image_v, cv2.MORPH_CLOSE, kernel_v_clean)

        sum = cv2.bitwise_or(close_image_v, close_image_h, mask = None) 


        out =  cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel_v_clean)

    return out
    

#rho	Distance resolution of the accumulator in pixels.
#theta	Angle resolution of the accumulator in radians.
#threshold	Accumulator threshold parameter. Only those lines are returned that get enough votes ( >threshold ).

def hough_op(	image, rho, theta, threshold):

    print(type(image))
    print((len(image)))

    imag2 = image.astype(np.uint8)

    lines	= cv2.HoughLines(	imag2.copy(), rho, theta, threshold)
    #lines =  HoughLinesP(imag2.copy(), 1, CV_PI/180, 50, 50, 10 )

    print("longueru ligen", np.shape(lines))

    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

   
        cv2.line(imag2,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imwrite('houghlines3.jpg',imag2)
   
    print( lines) 

    return lines



def get_kernel(width, angle):

    ker = np.ones((width*3,width),np.uint8)

    rot_ker = ndimage.rotate(ker, 180-angle, reshape=True)


    #return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(width+1, width+1))

    return rot_ker


def generate_output(s, filename, image_idx, all_nodes): 
    
    
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