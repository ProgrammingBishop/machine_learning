# Imports
# ==================================================
import os
import math
import warnings


# Default Libraries
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns

# Preprocessing
from sklearn.preprocessing   import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error
from scipy.special           import boxcox1p

# Model Selection
from sklearn.model_selection import RandomizedSearchCV, KFold
from xgboost                 import XGBRegressor
from sklearn.ensemble        import GradientBoostingRegressor
from sklearn.linear_model    import Lasso, ElasticNet
from sklearn.kernel_ridge    import KernelRidge
from mlxtend.regressor       import StackingCVRegressor
from keras.layers            import Dense, Dropout
from keras.models            import Sequential

import lightgbm as lgb


# Global Functions
# ==================================================
def get_numerical_features( df ):
    return df.select_dtypes( include = [ 'int64', 'float64' ] ).columns

def get_categorical_features( df ):
    return df.select_dtypes( include = [ 'object' ] ).columns

def return_features_with_null( df ):
    still_missing = pd.DataFrame( len( df[ get_categorical_features ] ) - df[ get_categorical_features ].count() )
    return pd.DataFrame( still_missing[ still_missing[ 0 ] > 0 ] )


# Load Data
# ==================================================
data_path = os.getcwd() + '\\..\\..\\data\\housing_prices\\'
train_raw = pd.read_csv( data_path + 'train.csv' )
test_raw  = pd.read_csv( data_path + 'test.csv' )


# Correct Outliers
# ==================================================
# Kaggle suggests to remove some outliers for Garage Living Area

# sns.distplot( x = train_raw[ 'GrLivArea' ], 
#               y = train_raw[ 'SalePrice' ] )
# plt.show()

train_raw = train_raw.drop( train_raw[ ( train_raw[ 'GrLivArea' ] > 4000 ) & ( train_raw[ 'SalePrice' ] < 250000 ) ].index )

train_X = train_raw.drop( [ 'SalePrice', 'Id' ], axis = 1 )
train_y = train_raw[ 'SalePrice' ]

test_X  = test_raw.drop( [ 'Id' ], axis = 1 )


# Transform Response
# ==================================================
print( 'Skew: {} \nKurtosis: {}'.format( round( train_y.skew(), 4 ), 
                                         round( train_y.kurtosis(), 4 ) ) )

train_y = np.log1p( train_y )

print( 'Skew: {} \nKurtosis: {}'.format( round( train_y.skew(), 4 ), 
                                         round( train_y.kurtosis(), 4 ) ) )

full_X    = pd.concat( [train_X, test_X] )
train_end = len( train_X )
test_end  = len( full_X )

print( full_X.shape )


# Data Preprocessing
# ==================================================
# Replace Missing
fill_with_none = [ 'Alley', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'Fence', 'FireplaceQu', 
                   'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'MasVnrType', 'MiscFeature', 'MSSubClass', 'PoolQC' ]

fill_with_zero = [ 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 
                   'GarageArea', 'GarageCars', 'TotalBsmtSF', 'GarageYrBlt', 'MasVnrArea' ]

full_X[ fill_with_none ] = full_X[ fill_with_none ].fillna( 'None' )
full_X[ fill_with_zero ] = full_X[ fill_with_zero ].fillna( 0 )

# Drop Useless Feature
print( len( full_X[ full_X[ 'Utilities' ] == 'AllPub' ] ) / len( full_X ) )

full_X = full_X.drop( [ 'Utilities' ], axis = 1 )

# Impute NaN
full_X[ 'Functional' ]      = full_X["Functional"].fillna( 'Typ' )

missing_with_mode           = [ 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'MSZoning' ]
full_X[ missing_with_mode ] = full_X[ missing_with_mode ].fillna( full_X.mode().iloc[0] )
full_X[ 'LotFrontage' ]     = full_X.groupby( 'Neighborhood' )[ 'LotFrontage' ].transform( lambda x : x.fillna( x.median() ) )

# Stringify
full_X[ 'MSSubClass' ]  = full_X['MSSubClass'].apply(str)
full_X[ 'OverallCond' ] = full_X['OverallCond'].astype(str)
full_X[ 'YrSold' ]      = full_X['YrSold'].astype(str)
full_X[ 'MoSold' ]      = full_X['MoSold'].astype(str)

# Ranked
full_X[ 'Alley'        ].replace( { 'None' : 0, 'Grvl' : 1, 'Pave' : 2 }, inplace = True )
full_X[ 'BsmtCond'     ].replace( { 'None' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4 }, inplace = True )
full_X[ 'BsmtExposure' ].replace( { 'None' : 0, 'No' : 1, 'Mn' : 2, 'Av' : 3, 'Gd' : 4 }, inplace = True )
full_X[ 'BsmtFinType1' ].replace( { 'None' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6 }, inplace = True )
full_X[ 'BsmtFinType2' ].replace( { 'None' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6 }, inplace = True )
full_X[ 'BsmtQual'     ].replace( { 'None' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 }, inplace = True )
full_X[ 'ExterCond'    ].replace( { 'None' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 }, inplace = True )
full_X[ 'ExterQual'    ].replace( { 'None' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 }, inplace = True )
full_X[ 'Fence'        ].replace( { 'None' : 0, 'MnWw' : 1, 'GdWo' : 2, 'MnPrv' : 3, 'GdPrv' : 4 }, inplace = True )
full_X[ 'FireplaceQu'  ].replace( { 'None' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 }, inplace = True )
full_X[ 'Functional'   ].replace( { 'None' : 0, 'Sal' : 1, 'Sev' : 2, 'Maj2' : 3, 'Maj1' : 4, 'Mod' : 5, 'Min2' : 6, 'Min1' : 7, 'Typ' : 8 }, inplace = True )
full_X[ 'GarageCond'   ].replace( { 'None' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 }, inplace = True )
full_X[ 'GarageFinish' ].replace( { 'None' : 0, 'Unf' : 1, 'RFn' : 2, 'Fin' : 3 }, inplace = True )
full_X[ 'GarageQual'   ].replace( { 'None' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 }, inplace = True )
full_X[ 'HeatingQC'    ].replace( { 'None' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 }, inplace = True )
full_X[ 'KitchenQual'  ].replace( { 'None' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 }, inplace = True )
full_X[ 'LandSlope'    ].replace( { 'None' : 0, 'Sev' : 1, 'Mod' : 2, 'Gtl' : 3 }, inplace = True )
full_X[ 'LandContour'  ].replace( { 'None' : 0, 'Low' : 1, 'HLS' : 2, 'Bnk' : 3, 'Lvl' : 4 }, inplace = True )
full_X[ 'LotShape'     ].replace( { 'None' : 0, 'Reg' : 1, 'IR1' : 2, 'IR2' : 3, 'IR3' : 4 }, inplace = True )
full_X[ 'PoolQC'       ].replace( { 'None' : 0, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 }, inplace = True )
full_X[ 'PavedDrive'   ].replace( { 'None' : 0, 'N' : 1, 'P' : 2, 'Y' : 3 }, inplace = True )

# Add New Feature
full_X[ 'TotalLivAreaSF' ] = full_X[ '1stFlrSF' ] + full_X[ '2ndFlrSF' ] + full_X[ 'TotalBsmtSF' ]

# BoxCox Transformation
categorical_features = list( get_categorical_features( full_X ) )
numerical_features   = list( get_numerical_features( full_X ) )
skew_features        = {}

for feature in numerical_features:
    skew_features[ feature ] = full_X[ feature ].skew()
    
skew_features = pd.DataFrame( { 'Features' : list( skew_features.keys() ), 
                                'Skew'     : list( skew_features.values() ) } )

features_to_box = list( skew_features[ abs( skew_features[ 'Skew' ] ) > 0.75 ][ 'Features' ] )

for feature in features_to_box:
    full_X[ [feature] ] = boxcox1p( full_X[ [feature] ], 0.15 )

# One-Hot-Encode
full_X = pd.get_dummies( full_X, 
                         drop_first = True, 
                         prefix     = categorical_features, 
                         columns    = categorical_features )

# Split Train/Test
train_X = pd.DataFrame( full_X[ 0:train_end ] )
test_X  = pd.DataFrame( full_X[ train_end:test_end ] )

print( "Train: {} \nTest: {}".format( train_X.shape, test_X.shape ) )


# Find Optimal Hyperparameters / Score Models
# ==================================================
scaler         = RobustScaler()
train_X_scaled = scaler.fit_transform( train_X )

rmse_score = {
    'XGBRegressor'              : 0,
    'GradientBoostingRegressor' : 0,
    'Lasso'                     : 0,
    'KernelRidge'               : 0,
    'ElasticNet'                : 0,
    'LightGBM'                  : 0,
    'NeuralNetwork'             : 0
}

# XGBRegressor
xgbm_grid = { 
    'colsample_bytree' : [ 0.25, 0.5, 0.75, 0.99 ], 
    'gamma'            : [ 0.001, 0.01, 0.05, 0.1, 0.15, 0.2 ], 
    'min_child_weight' : [ 1, 1.25, 1.5, 1.75, 2 ], 
    'n_estimators'     : [ 1000, 1500, 2000, 2500, 3000 ],
    'reg_alpha'        : [ 0, 0.25, 0.5, 0.75, 0.99 ], 
    'reg_lambda'       : [ 0, 0.25, 0.5, 0.75, 0.99 ],
    'subsample'        : [ 0.25, 0.5, 0.75, 0.99 ]
}

xgbm = RandomizedSearchCV( XGBRegressor(), cv = 5, param_grid = xgbm_grid, n_jobs = -1, scoring = 'neg_mean_squared_error', n_iter = 15, verbose = 1 )
xgbm.fit( train_X, train_y )

rmse_score[ 'XGBRegressor' ] = round( math.sqrt( -xgbm.best_score_ ), 4 )

# GradientBoostingRegressor
gbr_grid = {
    'n_estimators'      : [ 1000, 1500, 2000, 2500, 3000 ],
    'learning_rate'     : [ 0.001, 0.01, 0.05, 0.1, 0.15, 0.2 ],
    'max_depth'         : [ 3, 4, 5, 6, 7, 8 ],
    'max_features'      : [ 'sqrt', 'log2' ],
    'min_samples_leaf'  : [ 5, 7, 10, 12, 15 ], 
    'min_samples_split' : [ 5, 7, 10, 12, 15 ], 
    'loss'              : [ 'huber' ]
}

gbr = RandomizedSearchCV( GradientBoostingRegressor(), cv = 5, param_grid = gbr_grid, n_jobs = -1, scoring = 'neg_mean_squared_error', n_iter = 15, verbose = 1 )
gbr.fit( X_train, y_train )

rmse_score[ 'GradientBoostingRegressor' ] = round( math.sqrt( -gbr.best_score_ ), 4 )

# Lasso
lasso_grid = { 
    'alpha'    : [ 0.0001, 0.0005, 0.001, 0.0015, 0.002, 0.01, 0.1, 1.0 ], 
    'max_iter' : [ 250, 500, 1000, 2000, 3500, 5000 ]
}

lasso = RandomizedSearchCV( Lasso(), cv = 5, param_grid = lasso_grid, n_jobs = -1, scoring = 'neg_mean_squared_error', n_iter = 15, verbose = 1 )
lasso.fit( train_X_scaled, train_y )

rmse_score[ 'Lasso' ] = round( math.sqrt( -lasso.best_score_ ), 4 )

# KernelRidge
kernel_grid = {
    'alpha'  : [ 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ], 
    'kernel' : [ 'polynomial' ], 
    'degree' : [ 2, 3 ], 
    'coef0'  : [ 1, 1.5, 2, 2.5, 3 ]
}

kridge = RandomizedSearchCV( KernelRidge(), cv = 5, param_grid = kernel_grid, n_jobs = -1, scoring = 'neg_mean_squared_error', n_iter = 15, verbose = 1 )
kridge.fit( train_X, y_train )

rmse_score[ 'KernelRidge' ] = round( math.sqrt( -kridge.best_score_ ), 4 )

# Elastic Net
elastic_grid = {
    'alpha'    : [ 0.0001, 0.0005, 0.001, 0.0015, 0.002, 0.01, 0.1, 1.0 ],
    'l1_ratio' : [ 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]
}

elastic = RandomizedSearchCV( ElasticNet(), cv = 5, param_grid = elastic_grid, n_jobs = -1, scoring = 'neg_mean_squared_error', n_iter = 15, verbose = 1 )
elastic.fit( train_X_scaled, train_y )

rmse_score[ 'ElasticNet' ] = round( math.sqrt( -elastic.best_score_ ), 4 )

# LightGBM
lgbm_grid = {
    'objective'               : [ 'regression' ],
    'num_leaves'              : [ 2, 3, 4, 5, 6 ],
    'learning_rate'           : [ 0.001, 0.01, 0.05, 0.1, 0.15, 0.2 ], 
    'n_estimators'            : [ 250, 500, 1000, 2000, 3500, 5000 ],
    'max_bin'                 : [ 30, 40, 50, 60, 70 ],
    'bagging_fraction'        : [ 0.25, 0.5, 0.75, 0.99 ],
    'bagging_freq'            : [ 1, 3, 5, 7, 9 ], 
    'feature_fraction'        : [ 0.25, 0.5, 0.75, 0.99 ]
    'min_data_in_leaf'        : [ 1, 3, 5, 7, 9 ], 
    'min_sum_hessian_in_leaf' : [ 5, 7, 10, 12, 15 ]
}

lgbm = RandomizedSearchCV( lgb.LGBMRegressor(), cv = 5, param_grid = lgbm_grid, n_jobs = -1, scoring = 'neg_mean_squared_error', n_iter = 15, verbose = 1 )
lgbm.fit( train_X, train_y )

rmse_score[ 'LightGBM' ] = round( math.sqrt( -lgbm.best_score_ ), 4 )

# NeuralNetwork
def create_nueral_net():
    ann_model = Sequential()
    
    return

ann_grid = {
    
}

ann = RandomizedSearchCV( create_nueral_net(), cv = 5, param_grid = ann_grid, n_jobs = -1, scoring = 'neg_mean_squared_error', n_iter = 15, verbose = 1 )
ann.fit( train_X_scaled, train_y )

rmse_score[ 'NeuralNetwork' ] = round( math.sqrt( -ann.best_score_ ), 4 )


# Review RMSE Scores
# ==================================================
print( pd.DataFrame( { 'RMSE' : rmse_score }, index = rmse_score.keys() ).sort_values( by = [ 'RMSE' ], ascending = True ) )


# Build Stacked Ensemble Model
# ==================================================
stacked_model = StackingCVRegressor( regressors     = ( lasso_model, gbr_model, kridge_model, lgbm_model, xgbm_model ),
                                     meta_regressor = elastic_model,
                                     cv = 5 )

stacked_model.fit( train_X.values, train_y.values )
stacked_predict = np.expm1( stacked_model.predict( test_X.values ) )


# Generate Submission
# ==================================================
submission = pd.DataFrame(
    { 'Id'        : test_raw[ 'Id' ],
      'SalePrice' : stacked_predict } 
)

submission.to_csv( '.\\ames_housing_submission.csv', index = False )
print( submission.head(10) )
