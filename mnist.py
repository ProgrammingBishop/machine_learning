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
from dask.diagnostics        import ProgressBar

# Model Building
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.image   import ImageDataGenerator
from keras.utils.np_utils        import to_categorical # One-Hot-Encode Labels
from keras.models                import Sequential
from keras.optimizers            import RMSprop, Adam
from keras.layers                import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.callbacks             import ReduceLROnPlateau


# Global Functions
# ==================================================
def show_images( start_obs, df ):
    '''
    start_obs : Index to start for digit display
    df        : Pandas dataframe start_obs indexes from
    '''
    for index, _ in enumerate( range( start_obs, ( start_obs + 9 ) ) ):
        plt.subplot( 330 + 1 + index )
        plt.imshow( np.matrix( df[ index ] ), 
                    cmap = plt.get_cmap( 'gray' ) ) 
    plt.show()


def return_flow_batch( X, y ):
    '''
    X : Feature values
    y : Response values
    '''
    for X_batch, _ in datagen.flow( X, y, batch_size = 9 ):
        for image in range( 0, 9 ):
            plt.subplot( 330 + 1 + image )
            plt.imshow( X_batch[ image ].reshape( 28, 28 ), 
                        cmap = plt.get_cmap( 'gray' ) )
        plt.show()
        break


def return_nan_count( df ):
    '''
    df : DataFrame to check null values for
    '''
    print( df.isnull().any().describe() )


def return_keras_grid_results( grid_fit ):
    '''
    grid_fit : Keras object obtained through grid search
    '''
    mean       = grid_fit.cv_results_[ 'mean_test_score' ]
    std_dev    = grid_fit.cv_results_[ 'std_test_score' ]
    parameters = grid_fit.cv_results_[ 'params' ]

    for mean, std_dev, parameters in zip( parameters, mean, std_dev ):
        print("\
            Parameters : %r \n\
            Mean       : %f \n\
            STD        : %f \n"\
            % ( mean, std_dev, parameters ) 
        )


def return_keras_grid_parameters( grid_fit ):
    '''
    grid_fit : Keras object obtained through grid search
    '''
    mean        = grid_fit.cv_results_[ 'mean_test_score' ]
    best_mean   = mean.tolist().index( max( mean ) )
    best_params = grid_fit.cv_results_[ 'params' ][ best_mean ]

    print( best_params )

    return best_params


def plot_keras_history( 
    train_metric = 'acc',
    valid_metric = 'val_acc',
    eval_metric  = 'Accuracy'
):
    '''
    Plot Accuracy:
        train_metric: acc
        valid_metric: val_acc
    Plot Loss:
        train_metric: loss
        valid_metric: val_loss
    '''
    # Values to Plot
    plt.plot( history.history[ train_metric ] )
    plt.plot( history.history[ valid_metric ] )
    
    # Plot Labels
    plt.title( eval_metric )
    plt.ylabel( eval_metric )
    plt.xlabel( 'Epoch' )
    plt.legend( [ 'Train', 'Validation'], loc = 'best' )

    plt.show()


# Load Data
# ==================================================
# Full DataFrames
data_path = os.getcwd() + '\\..\\..\\data\\mnist\\'

# Train
full_train = pd.read_csv( data_path + 'train.csv' )
full_test  = pd.read_csv( data_path + 'test.csv' )

TEST_INDEX = pd.Series( full_test.index + 1 )

# Check for Missing
return_nan_count( full_train )
return_nan_count( full_test )

# Reshape 2D Matrix into 4D
train_X = full_train\
            .drop( [ 'label' ], axis = 1 )\
            .values.reshape( -1, 28, 28, 1 )

train_y = to_categorical( 
    y           = full_train[ 'label' ], 
    num_classes = len( set( full_train[ 'label' ] ) ) 
)

# Test
test_X = full_test\
            .values.reshape( -1, 28, 28, 1 )

# Remove Unnecessary Variables from Above
del full_train
del full_test

# Normalize Values
train_X = train_X / 255.0
test_X  = test_X  / 255.0

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
    zca_whitening                 = False,
    rotation_range                = 10,
    zoom_range                    = 0.1,
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
    optimizer       = 'RMSProp',
    kernel_size_one = 5,
    kernel_size_two = 3,
    strides         = 1,
    pool_size       = 2,
    dropout_one     = 0.25,
    dropout_two     = 0.50,
    lr              = 0.001, 
    rho             = 0.1,
    epsilon         = 1e-10,
    decay           = 0.1
):
    model = Sequential()

    model.add( 
        Conv2D( filters     = 32, 
                kernel_size = ( kernel_size_one, kernel_size_one ), 
                strides     = ( strides, strides ),
                padding     = 'same', 
                activation  = 'relu',
                input_shape = ( 28, 28, 1 ), 
                data_format = 'channels_last' 
        ) 
    )

    model.add( 
        Conv2D( filters     = 32, 
                kernel_size = ( kernel_size_one, kernel_size_one ), 
                strides     = ( strides, strides ),
                padding     = 'same',
                activation  = 'relu',
                data_format = 'channels_last' 
        ) 
    )

    model.add( 
        MaxPool2D( 
            pool_size = ( pool_size, pool_size ) 
        ) 
    )

    model.add(
        Dropout( dropout_one )
    )

    model.add( 
        Conv2D( 
            filters     = 64, 
            kernel_size = ( kernel_size_two, kernel_size_two ), 
            strides     = ( strides, strides ),
            padding     = 'same',
            activation  = 'relu', 
            data_format = 'channels_last' 
        ) 
    )

    model.add( 
        Conv2D( 
            filters     = 64, 
            kernel_size = ( kernel_size_two, kernel_size_two ), 
            strides     = ( strides, strides ),
            padding     = 'same',
            activation  = 'relu', 
            data_format = 'channels_last' 
        ) 
    )
    model.add( 
        MaxPool2D( 
            pool_size = ( pool_size, pool_size ),
            strides   = ( 2, 2 )
        ) 
    )

    model.add(
        Dropout( dropout_one )
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
        Dropout( dropout_two )
    )

    model.add( 
        Dense( 
            units      = 10, 
            activation = 'softmax' 
        ) 
    )

    optimizer = RMSprop(    
        lr      = lr, 
        rho     = rho, 
        epsilon = epsilon, 
        decay   = decay
    )

    model.compile(
        optimizer = optimizer, 
        loss      = "categorical_crossentropy", 
        metrics   = [ "accuracy" ]
    )

    return model


# Find Optimal Hyperparameters
# ==================================================
# After an Initial Grid Search RMSProp Performed Best
cnn_grid = {
    'optimizer'       : [ 'RMSProp' ],
    'kernel_size_one' : [ 3, 4, 5 ],
    'kernel_size_two' : [ 3, 4, 5 ],
    'dropout_one'     : [ 0.25, 0.5, 0.75 ],
    'dropout_two'     : [ 0.25, 0.5, 0.75 ],
    'lr'              : [ 0.0001, 0.001, 0.01 ], 
    'rho'             : [ 0.1, 0.3, 0.5, 0.7, 0.9 ],
    'epsilon'         : [ 1e-10, 1e-08, 1e-06 ], 
    'decay'           : [ 0.1, 0.3, 0.4, 0.6, 0.8 ]
}

mnist_classifier = KerasClassifier( build_fn = build_cnn, verbose = 1 )

cnn_model = RandomizedSearchCV(
    estimator           = mnist_classifier,
    param_distributions = cnn_grid,
    n_jobs              = 6,
    n_iter              = 10,
    cv                  = 5,
    verbose             = 1
)

with ProgressBar():
    cnn_model.fit( X_train, y_train )

# Return Results & Best Hyperparameters
return_keras_grid_results( cnn_model )
best_estimator = return_keras_grid_parameters( cnn_model )

# Rerun CNN with Best Hyperparameters
cnn_model = build_cnn( optimizer = best_estimator[ 'optimizer' ] )


# Fit CNN on MNIST & Evaluate Performance
# ==================================================
BATCH_SIZE = 86
EPOCHS     = 1
LEARN_RATE = ReduceLROnPlateau(
    monitor  = 'val_acc', 
    factor   = 0.1, 
    patience = math.ceil( EPOCHS / 10 ) + 1, 
    min_lr   = 0.00001,
    verbose  = 1, 
)

history = cnn_model.fit_generator(
    datagen.flow( 
        X_train, y_train, 
        batch_size = BATCH_SIZE
    ),
    epochs          = EPOCHS, 
    workers         = 6,
    validation_data = ( X_valid, y_valid ),
    verbose         = 1, 
    steps_per_epoch = math.ceil( X_train.shape[0] / BATCH_SIZE ), 
    callbacks       = [ LEARN_RATE ]
)


# Visualize Accuracy and Loss Performance
# ==================================================
# Accuracy
plot_keras_history( 'acc', 'val_acc' )

# Loss
plot_keras_history( 'loss', 'val_loss' )


# Get Predictions
# ==================================================
results = cnn_model.predict( test_X )
results = np.argmax( results, axis = 1 )
results = pd.Series( results, name = "Label" )


# Generate Submission
# ==================================================
submission = pd.DataFrame({ 
    'ImageId' : TEST_INDEX,
    'Label'   : results 
})

submission.to_csv( '.\\mnist_submission.csv', index = False )
print( submission.head(10) )