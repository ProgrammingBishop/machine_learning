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
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.image   import ImageDataGenerator
from keras.utils.np_utils        import to_categorical # One-Hot-Encode Labels
from keras.models                import Sequential
from keras.optimizers            import RMSprop, Adam
from keras.layers                import Conv2D, MaxPool2D, Flatten, Dense, Dropout


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

# Normalize Values
full_train = full_train / 255.0
full_test  = full_test  / 255.0

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
    featurewise_center            = False, 
    featurewise_std_normalization = False,
    samplewise_std_normalization  = False,  
    zca_whitening                 = True,
    rotation_range                = 90,
    zoom_range                    = 0.25,
    width_shift_range             = 0.1,
    height_shift_range            = 0.1,
    horizontal_flip               = False,
    vertical_flip                 = False
)  


datagen.fit( X_train )

# Preview Image Augmentations
return_flow_batch( X_train, y_train )


# Construct Convolutional Neural Network
# ==================================================
# Build CNN Architecure
def build_cnn( 
    optimizer   = 'RMSProp',
    kernel_size = 5,
    strides     = 1
):
    '''
    optimizer: Either "RMSProp" or "Adam" (default is RMSProp)
    kernel_size: Window size for convolutional layer (default is 5)
    strides: Shift amount for the kernel window (default is 1)
    '''
    model = Sequential()

    model.add( 
        Conv2D( filters     = 32, 
                kernel_size = ( 5, 5 ), 
                strides     = ( 1, 1 ),
                padding     = 'same',
                input_shape = ( 28, 28, 1 ), 
                activation  = 'relu', 
                data_format = 'channels_last' 
        ) 
    )

    model.add( 
        Conv2D( filters     = 32, 
                kernel_size = ( 5, 5 ), 
                strides     = ( 1, 1 ),
                padding     = 'same',
                activation  = 'relu'
        ) 
    )

    model.add( 
        MaxPool2D( pool_size = ( 2, 2 ) ) 
    )

    model.add(
        Dropout( 0.25 )
    )

    model.add( 
        Conv2D( 
            filters     = 64, 
            kernel_size = ( 3, 3 ), 
            strides     = ( 1, 1 ),
            padding     = 'same',
            activation  = 'relu', 
            data_format = 'channels_last' 
        ) 
    )

    model.add( 
        Conv2D( 
            filters     = 64, 
            kernel_size = ( 3, 3 ), 
            strides     = ( 1, 1 ),
            padding     = 'same',
            activation  = 'relu', 
            data_format = 'channels_last' 
        ) 
    )
    model.add( 
        MaxPool2D( 
            pool_size = ( 2, 2 ),
            strides   = ( 2, 2 )
        ) 
    )

    model.add(
        Dropout( 0.25 )
    )

    model.add( 
        Flatten() 
    )

    model.add( 
        Dense( 
            units      = 256, 
            activation = 'relu' 
        ) 
    )

    model.add(
        Dropout( 0.50 )
    )

    model.add( 
        Dense( 
            units      = 1, 
            activation = 'softmax' 
        ) 
    )

    # Dynamic Optimizer
    if optimizer == 'RMSProp':
        optimizer = RMSprop(    
            lr      = 0.001, 
            rho     = 0.9, 
            epsilon = 1e-08, 
            decay   = 0.0
        )
    
    elif optimizer == 'Adam':
        optimizer = Adam(    
            lr     = 0.001,
            beta_1 = 0.2,
            beta_2 = 0.7
        )

    else:
        print( 'Input "RMSProp" or "Adam"' )

    model.compile(
        optimizer = optimizer, 
        loss      = "categorical_crossentropy", 
        metrics   = [ "accuracy" ]
    )

    return model

Adam()

# Execute CNN & Get Results
cnn_grid = {
    'optimizer' : [ 'RMSProp', 'Adam' ]
}

mnist_classifier = KerasClassifier( build_fn = build_cnn, verbose = 0 )

cnn_model_grid = RandomizedSearchCV(
    estimator           = mnist_classifier,
    param_distributions = cnn_grid,
    n_jobs              = 6,
    cv                  = 5
)

cnn_model_fit = cnn_model_grid.fit( train_X, train_y )

# history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
#                               epochs = epochs, validation_data = (X_val,Y_val),
#                               verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
#                               , callbacks=[learning_rate_reduction])


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