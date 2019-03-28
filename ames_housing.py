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
from sklearn.metrics         import mean_squared_error
from scipy.special           import boxcox1p

# Model Selection
from sklearn.model_selection     import RandomizedSearchCV, KFold
from sklearn.pipeline            import Pipeline
from sklearn.base                import BaseEstimator, TransformerMixin, RegressorMixin, clone
from xgboost                     import XGBRegressor
from sklearn.ensemble            import GradientBoostingRegressor
from sklearn.linear_model        import Lasso, ElasticNet
from sklearn.kernel_ridge        import KernelRidge

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
full_X[ 'TotalPorch' ]     = full_X[ 'OpenPorchSF' ] + full_X[ 'EnclosedPorch' ] + full_X[ '3SsnPorch' ] + full_X[ 'ScreenPorch' ]
full_X[ 'TotalBath' ]      = full_X[ 'BsmtFullBath' ] + ( full_X[ 'BsmtHalfBath' ] * 0.5 ) + full_X[ 'FullBath' ] + ( full_X[ 'HalfBath' ] * 0.5 )

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
rmse_score = {
    'XGBRegressor'              : 0,
    'GradientBoostingRegressor' : 0,
    'Lasso'                     : 0,
    'KernelRidge'               : 0,
    'ElasticNet'                : 0,
    'LightGBM'                  : 0
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

xgbm = RandomizedSearchCV( 
    XGBRegressor(), 
    cv                  = 5, 
    param_distributions = xgbm_grid, 
    n_jobs              = -1, 
    scoring             = 'neg_mean_squared_error', 
    n_iter              = 20, 
    verbose             = 1 
)

xgbm.fit( train_X, train_y )

rmse_score[ 'XGBRegressor' ] = round( math.sqrt( -xgbm.best_score_ ), 4 )

# GradientBoostingRegressor
gbr_grid = {
    'n_estimators'      : [ 2000, 2500, 3000 ],
    'learning_rate'     : [ 0.05, 0.1, 0.2, 0.3 ],
    'max_depth'         : [ 3, 4, 5, 6, ],
    'max_features'      : [ 'sqrt', 'log2' ],
    'min_samples_leaf'  : [ 5, 7, 10, 12, 15 ], 
    'min_samples_split' : [ 5, 7, 10, 12, 15 ], 
    'loss'              : [ 'huber' ]
}

gbr = RandomizedSearchCV( 
    GradientBoostingRegressor(), 
    cv                  = 5, 
    param_distributions = gbr_grid, 
    n_jobs              = -1, 
    scoring             = 'neg_mean_squared_error', 
    n_iter              = 20, 
    verbose             = 1 
)

gbr.fit( train_X, train_y )

rmse_score[ 'GradientBoostingRegressor' ] = round( math.sqrt( -gbr.best_score_ ), 4 )

# Lasso
lasso_grid = { 
    'lasso__alpha'    : [ 0.0001, 0.0005, 0.001, 0.0015, 0.002, 0.01, 0.1, 1.0 ], 
    'lasso__max_iter' : [ 250, 500, 1000, 2000, 3500, 5000 ]
}

lasso_steps    = [ ( 'scaler_robust', RobustScaler() ), ( 'lasso', Lasso() ) ]
lasso_pipeline = Pipeline( lasso_steps )

lasso = RandomizedSearchCV( 
    lasso_pipeline,
    param_distributions = lasso_grid,
    cv                  = 5,
    n_jobs              = -1,
    scoring             = 'neg_mean_squared_error',
    n_iter              = 100, 
    verbose             = 1 
)

lasso.fit( train_X, train_y )

rmse_score[ 'Lasso' ] = round( math.sqrt( -lasso.best_score_ ), 4 )

# KernelRidge
kernel_grid = {
    'kridge__alpha'  : [ 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ], 
    'kridge__kernel' : [ 'polynomial' ], 
    'kridge__degree' : [ 2 ], 
    'kridge__coef0'  : [ 5.0, 7.5, 10.0, 12.5, 15.0 ]
}

kridge_steps    = [ ( 'scaler_robust', RobustScaler() ), ( 'kridge', KernelRidge() ) ]
kridge_pipeline = Pipeline( kridge_steps )

kridge = RandomizedSearchCV( 
    kridge_pipeline,
    param_distributions = kernel_grid,
    cv                  = 5,
    n_jobs              = -1,
    scoring             = 'neg_mean_squared_error',
    n_iter              = 30, 
    verbose             = 1
)

kridge.fit( train_X, train_y )

rmse_score[ 'KernelRidge' ] = round( math.sqrt( -kridge.best_score_ ), 4 )

# Elastic Net
elastic_grid = {
    'elastic__alpha'    : [ 0.001, 0.0015, 0.002, 0.01, 0.1, 1.0 ],
    'elastic__l1_ratio' : [ 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]
}

elastic_steps    = [ ( 'scaler_robust', RobustScaler() ), ( 'elastic', ElasticNet() ) ]
elastic_pipeline = Pipeline( elastic_steps )

elastic = RandomizedSearchCV( 
    elastic_pipeline,
    param_distributions = elastic_grid,
    cv                  = 5,
    n_jobs              = -1,
    scoring             = 'neg_mean_squared_error',
    n_iter              = 30, 
    verbose             = 1
)

elastic.fit( train_X, train_y )

rmse_score[ 'ElasticNet' ] = round( math.sqrt( -elastic.best_score_ ), 4 )

# LightGBM
lgbm_grid = {
    'objective'               : [ 'regression' ],
    'num_leaves'              : [ 2, 3, 4 ],
    'learning_rate'           : [ 0.005, 0.01, 0.015 ], 
    'n_estimators'            : [ 500, 1000, 1500 ],
    'max_bin'                 : [ 80, 90, 100 ],
    'bagging_fraction'        : [ 0.65, 0.75, 0.85 ],
    'bagging_freq'            : [ 7 ], 
    'feature_fraction'        : [ 0.25, 0.5 ],
    'min_data_in_leaf'        : [ 3, 4, 5, 7, 9 ], 
    'min_sum_hessian_in_leaf' : [ 8, 10, 12 ],
    'colsample_bytree'        : [ 0.25, 0.5, 0.75, 1.0 ]
}

lgbm = RandomizedSearchCV( 
    lgb.LGBMRegressor(), 
    cv                  = 5, 
    param_distributions = lgbm_grid, 
    n_jobs              = -1, 
    scoring             = 'neg_mean_squared_error', 
    n_iter              = 20, 
    verbose             = 1 
)

lgbm.fit( train_X, train_y )

rmse_score[ 'LightGBM' ] = round( math.sqrt( -lgbm.best_score_ ), 4 )


# Review RMSE Scores
# ==================================================
print( pd.DataFrame( { 'RMSE' : rmse_score }, index = rmse_score.keys() ).sort_values( by = [ 'RMSE' ], ascending = True ) )


# Build Stacked Ensemble Model
# ==================================================
# XGBRegressor
xgbm_model = XGBRegressor(
    colsample_bytree = xgbm.best_estimator_.colsample_bytree,
    gamma            = xgbm.best_estimator_.gamma,
    min_child_weight = xgbm.best_estimator_.min_child_weight,
    n_estimators     = xgbm.best_estimator_.n_estimators,
    reg_alpha        = xgbm.best_estimator_.reg_alpha,
    reg_lambda       = xgbm.best_estimator_.reg_lambda,
    subsample        = xgbm.best_estimator_.subsample
)

# GradientBoostingRegressor
gbr_model = GradientBoostingRegressor( 
    n_estimators      = gbr.best_estimator_.n_estimators,
    learning_rate     = gbr.best_estimator_.learning_rate,
    max_depth         = gbr.best_estimator_.max_depth,
    max_features      = gbr.best_estimator_.max_features,
    min_samples_leaf  = gbr.best_estimator_.min_samples_leaf,
    min_samples_split = gbr.best_estimator_.min_samples_split,
    loss              = gbr.best_estimator_.loss
)

# Lasso
lasso_steps = [ ( 'scaler_robust', RobustScaler() ), 
                ( 'lasso', Lasso( 
                    alpha    = lasso.best_estimator_.get_params()[ 'lasso__alpha' ],
                    max_iter = lasso.best_estimator_.get_params()[ 'lasso__max_iter' ]
                ) ) ]

lasso_pipeline = Pipeline( lasso_steps )

# KernelRidge
kridge_steps = [ ( 'scaler_robust', RobustScaler() ), 
                 ( 'kridge', KernelRidge( 
                    alpha  = kridge.best_estimator_.get_params()[ 'kridge__alpha' ],
                    kernel = kridge.best_estimator_.get_params()[ 'kridge__kernel' ],
                    degree = kridge.best_estimator_.get_params()[ 'kridge__degree' ],
                    coef0  = kridge.best_estimator_.get_params()[ 'kridge__coef0' ]
                 ) ) ]

kridge_pipeline = Pipeline( kridge_steps )

# ElasticNet
elastic_steps = [ ( 'scaler_robust', RobustScaler() ), 
                  ( 'elastic', ElasticNet( 
                        alpha    = elastic.best_estimator_.get_params()[ 'elastic__alpha' ],
                        l1_ratio = elastic.best_estimator_.get_params()[ 'elastic__l1_ratio' ]
                  ) ) ]

elastic_pipeline = Pipeline( elastic_steps )

# LightGBM
lgbm_model = lgb.LGBMRegressor(
    objective               = lgbm.best_estimator_.objective,
    num_leaves              = lgbm.best_estimator_.num_leaves,
    learning_rate           = lgbm.best_estimator_.learning_rate,
    n_estimators            = lgbm.best_estimator_.n_estimators,
    max_bin                 = lgbm.best_estimator_.max_bin,
    bagging_fraction        = lgbm.best_estimator_.bagging_fraction,
    bagging_freq            = lgbm.best_estimator_.bagging_freq,
    feature_fraction        = lgbm.best_estimator_.feature_fraction,
    min_data_in_leaf        = lgbm.best_estimator_.min_data_in_leaf,
    min_sum_hessian_in_leaf = lgbm.best_estimator_.min_sum_hessian_in_leaf,
    colsample_bytree        = lgbm.best_estimator_.colsample_bytree
) 


# Execute Stacked Ensemble Model
# ==================================================
# Stacked Ensemble Class Override
# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# The following class override performs slightly better than StackingCVRegressor
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__( self, base_models, meta_model, n_folds = 5 ):
        self.base_models = base_models
        self.meta_model  = meta_model
        self.n_folds     = n_folds
        
    def fit( self, X, y ):
        self.base_models_       = [ list() for x in self.base_models ]
        self.meta_model_        = clone( self.meta_model )
        kfold                   = KFold( n_splits = self.n_folds, shuffle = True, random_state = 156 )
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate( self.base_models ):
            for train_index, holdout_index in kfold.split( X, y ):
                instance = clone( model )

                self.base_models_[i].append( instance )
                instance.fit( X[ train_index ], y[ train_index ] )

                y_pred = instance.predict( X[ holdout_index ] )
                out_of_fold_predictions[ holdout_index, i ] = y_pred
                
        self.meta_model_.fit( out_of_fold_predictions, y )

        return self
   
    def predict( self, X ):
        meta_features = np.column_stack( [
            np.column_stack( [ model.predict(X) for model in base_models ] ).mean( axis = 1 )
            for base_models in self.base_models_ 
        ] )

        return self.meta_model_.predict( meta_features )

# Stacked Ensemble Model
stacked_model = StackingAveragedModels( base_models = ( elastic_pipeline, gbr_model, kridge_pipeline ),
                                        meta_model  = lasso_pipeline  )

stacked_model.fit( train_X.values, train_y.values )
xgbm_model.fit( train_X, train_y )
lgbm_model.fit( train_X, train_y )

stacked_predict = np.expm1( stacked_model.predict( test_X.values ) )
xgbm_predict    = np.expm1(    xgbm_model.predict( test_X ) )
lgbm_predict    = np.expm1(    lgbm_model.predict( test_X ) )

# Ensemble Model Averaging
ensemble = ( 
      stacked_predict * 0.70
    + xgbm_predict    * 0.175
    + lgbm_predict    * 0.125
)


# Generate Submission
# ==================================================
submission = pd.DataFrame(
    { 'Id'        : test_raw[ 'Id' ],
      'SalePrice' : ensemble } 
)

# Ranked in Top 24%
submission.to_csv( '.\\ames_housing_submission.csv', index = False )
print( submission.head(10) )