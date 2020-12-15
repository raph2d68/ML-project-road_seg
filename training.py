import tensorflow as tf
import os

from pre_processing import *
from unet import *
from helper import *
from parameters import *

#MODIFY IF NECESSARY
PATH_TRAIN_DIR = 'ressource_files/training/'
PATH_UNET_MODEL = 'model_files/unet.h5'

def main(argv=None):
    train_image_dir = PATH_TRAIN_DIR + "images/"
    train_grth_dir = PATH_TRAIN_DIR + "groundtruth/"
    
    chosen_img_names = get_data_names(train_image_dir, TRAINING_SIZE, RAND_TRAIN )
    imgs, grth_imgs = augment_data(train_image_dir, train_grth_dir, chosen_img_names)
    
    print('imgs shape', imgs.shape)
    print('grth shape', grth.shape)
    
    x_train = np.asarray(imgs)
    y_train = np.expand_dims(np.asarray(grth_imgs), axis=3)

    # Create Model - refer to parameters.py for details
    model = unet_model(IMG_SIZE, NUM_CHANNELS, NUM_FILTER, FILTER_SIZE, dropout=0.5)

    # Run Model
    model, f1_scores = train_model(model, x_train, y_train, BATCH_SIZE, NUM_EPOCHS, valid_split)

    # Save the trained model
    print('Saving trained model')
    model.save(PATH_UNET_MODEL)

    plot_metric_history(f1_scores)


if __name__ == '__main__':
    tf.compat.v1.app.run()


