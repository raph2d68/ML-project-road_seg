from helper import *
import os
import numpy as np
import matplotlib.image as mpimg
import re

PATH_PREDICTION_DIR = 'ressource_files/prediction/'
PATH_SUBMISSION = 'final_submission.csv'


foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def mask_to_submission_(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))

# Post-processing try, but wasn't better
#img_pred_pp = [PATH_PREDICTION_DIR + 'prediction_pp_' + str(i+1) + '.png' for i in range(50)]
#mask_to_submission_(PATH_SUBMISSION_pp, *img_pred_pp)

img_pred = [PATH_PREDICTION_DIR + 'prediction_' + str(i+1) + '.png' for i in range(50)]

mask_to_submission_(PATH_SUBMISSION, *img_pred)


            
            