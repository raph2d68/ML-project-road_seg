import tensorflow as tf
import tensorflow.keras as keras

from ressource_files.mask_to_submission import *
from helper import *
from post_processing import *



# Split 608 wide image into 4 smaller 400*400 images to be able to use the model
# Input:    img         608*608 input image 
#           model       Unet model     
# Output    pred        reconstructed 608*608 image from 4 400*400 images    
def img_predict(img, model):
    """0 and 1 matrix prediction from an image"""
    imgs = np.array([img[:400, :400], img[:400, -400:], img[-400:, :400], img[-400:, -400:]])

    preds = model.predict(imgs)
    pred = np.zeros((img.shape[0], img.shape[1], 1))
    pred[:400, :400]    = preds[0]
    pred[:400, -400:]   = preds[1]
    pred[-400:, :400]   = preds[2]
    pred[-400:, -400:]  = preds[3]
    return pred


def main(argv=None): 

    img_test = [PATH_TEST_DATA + 'test_' + str(i+1 ) + '/' + 'test_' + str(i + 1) + '.png' for i in range(50)]
    model = keras.models.load_model(PATH_model)

    for idx, path in enumerate(img_test):    
        print(idx)
        img = np.squeeze(get_image(path))
        pred = img_predict(img, model)
        pred = np.squeeze(pred).round()

        #generate prediction image, with overlay for for visual inspection
        generate_output(idx, img, pred)


if __name__ == '__main__':
    tf.compat.v1.app.run()
