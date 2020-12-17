NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 10
VALIDATION_SIZE = 10  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16  # 64
NUM_EPOCHS = 20
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
RECORDING_STEP = 0

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 20

POST_PROCESS = 1
TOL_ANGLE = 30
RAND_TRAIN = 0

#Unet-param
IMG_SIZE = 400
NUM_FILTER = 32
FILTER_SIZE = 3
NUMBER_OF_KERNEL = 3

#Path for training
TRAIN_DIR =         '/content/gdrive/MyDrive/ML-project2-main/ML-project2-main/ressource_files/training/'
PATH_UNET_MODEL =   '/content/gdrive/MyDrive/ML-project2-main/ML-project2-main/model_files/'

#Path for prediction 
PATH_model =        PATH_UNET_MODEL +'unet.h5'
PATH_TEST_DATA =    '/content/gdrive/MyDrive/ML-project2-main/ML-project2-main/ressource_files/test_set_images/'
PATH_PREDICTION_DIR = '/content/gdrive/MyDrive/ML-project2-main/ML-project2-main/ressource_files/prediction/'
PATH_SUBMISSION =   'final_submission.csv'
