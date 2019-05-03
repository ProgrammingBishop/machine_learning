# Imports
# ==================================================
from collections     import Counter, OrderedDict
from ast             import literal_eval
from Utilities       import Utilities
from sklearn.cluster import KMeans

import numpy             as np 
import pandas            as pd 
import matplotlib.pyplot as plt


# Class
# ==================================================
class TweepyKMeans():
    # PRIVATE
    __util = Utilities()


    def __following_to_sparse( self, c ):
        '''
        Returns : sparse matrix of binary feature values; 
        0 = not following user | 1 = following user
        --------------------------------------------------
        '''
        follower_data                = pd.read_csv( c.FOLLOWER_FRIENDS_CSV )
        columns_names                = pd.read_csv( c.TOP_FRIENDS_FOLLOWED_CSV )
        follower_data[ 'following' ] = follower_data[ 'following' ].apply( lambda ids : literal_eval( ids ) )
        friends                      = []

        for follower_friends in follower_data[ 'following' ]:
            friends += follower_friends

        features        = Counter( friends )
        sorted_features = OrderedDict( sorted( 
            features.items(), 
            key = lambda kv: 
            kv[1], 
            reverse = True
        ) )

        friends = list( sorted_features.keys() )[ 0:c.TOP_N ]
        friends = list( map( str, friends ) )

        sparse_matrix = pd.DataFrame( 
            index   = follower_data[ 'screen_name' ],
            columns = friends
        )

        sparse_matrix = sparse_matrix.fillna( 0 )

        for index, follower in follower_data.iterrows():
            for friend in follower[ 'following' ]:
                if str( friend ) in friends:
                    sparse_matrix.loc[ sparse_matrix.index[ index ], str( friend ) ] = 1

            if index % 100 == 0:
                self.__util.print_progress( index, len( follower_data ) )

        sparse_matrix.columns          = columns_names.loc[ columns_names.index[ 0:c.TOP_N ], 'screen_name' ]
        sparse_matrix[ 'screen_name' ] = list( follower_data[ 'screen_name' ] )

        self.__util.write_to_file( c.SPARSE_FRIENDS_MATRIX_CSV, pd.DataFrame( sparse_matrix ) )


    # PUBLIC
    def find_optimal_k( self, c ):
        '''
        Returns : find optimal number of clusters
        --------------------------------------------------
        '''
        self.__following_to_sparse( c )
        sparse_matrix = pd.read_csv( c.SPARSE_FRIENDS_MATRIX_CSV )
        sparse_matrix = sparse_matrix.drop( 'screen_name', axis = 1 )
        sum_squares   = []
        clusters      = range( 1, c.TOP_N, 10 )

        for k in clusters:
            k_means = KMeans( n_clusters = k )
            k_means = k_means.fit( sparse_matrix )
            sum_squares.append( k_means.inertia_ )

        plt.plot( clusters, sum_squares, 'go-' )
        plt.xlabel( 'Clusters' )
        plt.ylabel( 'Sum of Square Distances' )
        plt.title( 'Selection for Optimal K' )
        plt.show()

    
    # def create_cluster_labels( self, filepath, n_clusters ):
    #     '''
    #     Returns : CSV with row indexes follower screen_name and new column Cluster for labels
    #     --------------------------------------------------
    #     filepath   : location to obtain data
    #     n_clusters : number of centroids to plot 
    #     '''
    #     dataframe     = pd.read_csv( filepath )
    #     sparse_matrix = dataframe.drop( [ 'screen_name', 'Blizzard_Ent' ], axis = 1 )

    #     k_means = KMeans( n_clusters = n_clusters )
    #     k_means.fit( sparse_matrix )

    #     dataframe[ 'Cluster' ] = k_means.labels_
    #     dataframe.drop( 'Blizzard_Ent', axis = 1, inplace = True )
    #     dataframe.set_index( 'screen_name', inplace = True )

    #     self.util.write_to_file( c.SPARSE_MATRIX_WLABELS_CSV, pd.DataFrame( dataframe ), index = True )