import numpy             as np 
import pandas            as pd 
import matplotlib.pyplot as plt
import seaborn           as sns
import configurations    as c
import ast
import collections

from SaveToFile      import SaveToFile
from sklearn.cluster import KMeans


class ExploreMarketSegmentation():
    save = SaveToFile()

    def convert_to_sparse( self, filepath, top_n ):
        '''
        Returns : sparse matrix of binary variables; 0 = not following | 1 = following
        --------------------------------------------------
        filepath : location to obtain data
        top_n    : number of top most friended to plot (number + 1: upper-bound exclusive)
        '''
        follower_data                = pd.read_csv( filepath )
        columns_names                = pd.read_csv( c.TOP_FRIENDS_FOLLOWED_CSV )
        follower_data[ 'following' ] = follower_data[ 'following' ].apply( lambda ids : ast.literal_eval( ids ) )
        friends                      = []

        # Obtain Top Friends as Features
        for follower_friends in follower_data[ 'following' ]:
            friends += follower_friends

        features        = collections.Counter( friends )
        sorted_features = collections.OrderedDict( sorted( 
            features.items(), 
            key = lambda kv: 
            kv[1], 
            reverse = True
        ) )

        friends = list( sorted_features.keys() )[ 0:top_n ]
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
                print( "Progress: {}%"\
                    .format( str( round( index / len( follower_data ) * 100, 2 ) ) ) )

        sparse_matrix.columns          = columns_names.loc[ columns_names.index[ 0:151 ], 'screen_name' ]
        sparse_matrix[ 'screen_name' ] = list( follower_data[ 'screen_name' ] )

        self.save.write_to_csv_file( c.SPARSE_FRIENDS_MATRIX_CSV, pd.DataFrame( sparse_matrix ) )
        

    def find_optimal_k( self, filepath, top_n ):
        '''
        Returns : find optimal number of clusters
        --------------------------------------------------
        filepath : location to obtain data
        top_n    : number of top most friended to plot (number + 1: upper-bound exclusive)
        '''
        sparse_matrix = pd.read_csv( filepath )
        sparse_matrix = sparse_matrix.drop( 'screen_name', axis = 1 )
        sum_squares   = []
        clusters      = range( 1, top_n, 10 )

        for k in clusters:
            k_means = KMeans( n_clusters = k )
            k_means = k_means.fit( sparse_matrix )
            sum_squares.append( k_means.inertia_ )

        plt.plot( clusters, sum_squares, 'go-' )
        plt.xlabel( 'Clusters' )
        plt.ylabel( 'Sum of Square Distances' )
        plt.title( 'Selection for Optimal K' )
        plt.show()

    
    def create_cluster_labels( self, filepath, n_clusters ):
        '''
        Returns : CSV with row indexes follower screen_name and new column Cluster for labels
        --------------------------------------------------
        filepath   : location to obtain data
        n_clusters : number of centroids to plot 
        '''
        dataframe     = pd.read_csv( filepath )
        sparse_matrix = dataframe.drop( [ 'screen_name', 'Blizzard_Ent' ], axis = 1 )

        k_means = KMeans( n_clusters = n_clusters )
        k_means.fit( sparse_matrix )

        dataframe[ 'Cluster' ] = k_means.labels_
        dataframe.drop( 'Blizzard_Ent', axis = 1, inplace = True )
        dataframe.set_index( 'screen_name', inplace = True )

        self.save.write_to_csv_file( c.SPARSE_MATRIX_WLABELS_CSV, pd.DataFrame( dataframe ), index = True )