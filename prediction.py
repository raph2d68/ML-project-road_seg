import tensorflow as tf
from helper import *
import tensorflow.keras as keras
from ressource_files.mask_to_submission import *

PATH_ws = 'model_files/weights.h5'
PATH_model = 'model_files/unet.h5'
PATH_TEST_DATA = 'ressource_files/test_set_images/'
PATH_PREDICTION_DIR = 'ressource_files/predictions/'
PATH_SUBMISSION = 'final_submission.csv'

def predict_from_model(model, path_pred, *image_names):
    """Predict labels of test images from trained model and generate prediction as black and white image"""
    for idx, path in enumerate(image_filenames):
        img = np.squeeze(get_image(path))
        pred = img_predict(img, model)
        pred = np.squeeze(pred).round()
        pred = img_float_to_uint8(pred)
        pred_name = path_pred + 'pred_' + str(idx + 1) + '.png'
        Image.fromarray(pred).save(pred_name)

def img_predict(img, model):
    """0 and 1 matrix prediction from an image"""
    imgs = np.array([img[:400, :400], img[:400, -400:], img[-400:, :400], img[-400:, -400:]])
    preds = model.predict(imgs)

    pred = np.zeros((img.shape[0], img.shape[1], 1))

    pred[:400, :400] = preds[0]
    pred[:400, -400:] = preds[1]
    pred[-400:, :400] = preds[2]
    pred[-400:, -400:] = preds[3]

    return prediction

img_test = [PATH_TEST_DATA + 'test_' + str(i + 1) + '/' + 'test_' + str(i + 1) + '.png' for i in range(50)]
img_pred = [PATH_PREDICTION_DIR + 'pred_' + str(i + 1) + '_unet.png' for i in range(50)]
model = keras.models.load_model(PATH_model)
predict_from_model(model, PATH_PREDICTION_DIR, *img_test)

# Generates the submission
mask_to_submission(PATH_SUBMISSION, *img_pred)

            
            