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
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils      import to_categorical # One-Hot-Encode Labels


# Global Functions
# ==================================================
def show_images( start_obs, df ):
    '''
    start_obs: index to start for digit display
    df: pandas dataframe start_obs indexes from
    '''
    for index, image in enumerate( range( start_obs, ( start_obs + 9 ) ) ):
        plt.subplot( 330 + 1 + index )
        plt.imshow( np.matrix( df[ index ] ), 
                    cmap = plt.get_cmap( 'gray' ) ) 
    plt.show()


def return_flow_batch( X, y ):
    for X_batch, _ in datagen.flow( X, y, batch_size = 9 ):
        for image in range( 0, 9 ):
            plt.subplot( 330 + 1 + image )
            plt.imshow( X_batch[ image ].reshape( 28, 28 ), 
                        cmap = plt.get_cmap( 'gray' ) )
        plt.show()
        break


# Load Data
# ==================================================
# Full DataFrames
data_path = os.getcwd() + '\\..\\..\\data\\mnist\\'

# Train
full_train = pd.read_csv( data_path + 'train.csv' )
full_test  = pd.read_csv( data_path + 'test.csv' )

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


# Image Augmentation
# ==================================================
datagen = ImageDataGenerator(
    featurewise_center            = True, 
    featurewise_std_normalization = True,
    zca_whitening                 = True
)

datagen.fit( train_X )



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