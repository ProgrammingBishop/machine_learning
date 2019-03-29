# Imports
# ==================================================
import os
import math
import warnings

warnings.filterwarnings( 'ignore' )

# Default Libraries
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns

# Preprocessing
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils      import to_categorical # One-Hot-Encode Labels
from keras.models              import Sequential
from keras.layers              import Convolution2D, MaxPooling2D, Flatten, Dense


# Global Functions
# ==================================================
def show_images( start_obs, df ):
    '''
    start_obs: Index to start for digit display
    df: Pandas dataframe start_obs indexes from
    '''
    for index, _ in enumerate( range( start_obs, ( start_obs + 9 ) ) ):
        plt.subplot( 330 + 1 + index )
        plt.imshow( np.matrix( df[ index ] ), 
                    cmap = plt.get_cmap( 'gray' ) ) 
    plt.show()


def return_flow_batch( X, y ):
    '''
    X: Feature values
    y: Response values
    '''
    for X_batch, _ in datagen.flow( X, y, batch_size = 9 ):
        for image in range( 0, 9 ):
            plt.subplot( 330 + 1 + image )
            plt.imshow( X_batch[ image ].reshape( 28, 28 ), 
                        cmap = plt.get_cmap( 'gray' ) )
        plt.show()
        break


def return_nan_count( df ):
    print( df.isnull().any().describe() )


# Load Data
# ==================================================
# Full DataFrames
data_path = os.getcwd() + '\\..\\..\\data\\mnist\\'

# Train
full_train = pd.read_csv( data_path + 'train.csv' )
full_test  = pd.read_csv( data_path + 'test.csv' )

# Check for Missing
return_nan_count( full_train )
return_nan_count( full_test )

# Reshape 2D Matrix into 4D
train_X = full_train\
            .drop( [ 'label' ], axis = 1 )\
            .astype( 'float32' )\
            .values.reshape( -1, 28, 28, 1 )

train_y = to_categorical( 
    y           = full_train[ 'label' ], 
    num_classes = len( set( full_train[ 'label' ] ) ) 
)

# Test
test_X = full_test\
            .astype( 'float32' )\
            .values.reshape( -1, 28, 28, 1 )

# Preview Digits
show_images( 0, train_X )


# Train / Validation Split
# ==================================================
VALIDATION_SIZE = 0.2

X_train, X_valid, y_train, y_valid = train_test_split( 
    train_X, 
    train_y, 
    test_size    = VALIDATION_SIZE, 
    random_state = 19920917 
)


# Image Augmentation
# ==================================================
datagen = ImageDataGenerator(
    featurewise_center            = True, 
    featurewise_std_normalization = True,
    zca_whitening                 = True
)

datagen.fit( X_train )

# Preview Image Augmentations
return_flow_batch( X_train, y_train )


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