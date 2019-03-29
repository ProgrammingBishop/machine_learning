# Imports
# ==================================================
import os
import math
import warnings

warnings.filterwarnings("ignore")

# Default Libraries
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns

# Preprocessing
from keras.preprocessing.image import ImageDataGenerator


# Load Data
# ==================================================
data_path = os.getcwd() + '\\..\\..\\data\\mnist\\'
train_raw = data_path + 'train.csv'
test_raw  = data_path + 'test.csv'

test_datagen  = ImageDataGenerator( rescale = 1.0 / 255 )
train_datagen = ImageDataGenerator( rescale         = 1.0 / 255,
                                    shear_range     = 0.2,
                                    zoom_range      = 0.2,
                                    horizontal_flip = True )

train_set = train_datagen.flow_from_directory( train_raw,
                                               target_size = ( 64, 64 ),
                                               batch_size  = 32,
                                               class_mode  = 'binary' )

test_set = test_datagen.flow_from_directory( test_raw,
                                             target_size = ( 64, 64 ),
                                             batch_size  = 32,
                                             class_mode  = 'binary' )


# Generate Submission
# ==================================================
submission = pd.DataFrame(
    { 
        'ImageId' : None,
        'Label'   : None 
    } 
)

submission.to_csv( '.\\mnist_submission.csv', index = False )
print( submission.head(10) )