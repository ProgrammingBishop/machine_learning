import numpy             as np 
import pandas            as pd 
import matplotlib.pyplot as plt
import configurations    as c
import ast

from SaveToFile import SaveToFile

class ExploreMarketSegmentation():
    save = SaveToFile()

    def convert_to_sparse( self, filepath ):
        '''
        Returns : sparse matrix of binary variables; 0 = not following | 1 = following
        --------------------------------------------------
        filepath : location to obtain data
        '''
        follower_data = pd.read_csv( filepath )
        followed      = follower_data[ 'following' ].apply( lambda ids : ast.literal_eval( ids ) )
        features      = []

        # Obtain Set of Indexes
        for friend in followed:
            features += friend

        # Convert Friend IDs to Features
        column_names                  = set( features.sort()  )
        sparse_matrix                 = { 'ID' : follower_data[ 'screen_name' ] }
        sparse_matrix[ column_names ] = 0

        # For Every Follower
        for follower in follower_data:
            # For Every Feature in Follower's Friends
            for feature in follower[ 'following' ]:
                # Flip Feature if Follower is Friends
                if feature in column_names:
                    sparse_matrix[ feature ] = 1

        self.save.write_to_csv_file( c.SPARSE_FRIENDS_MATRIX_CSV, pd.DataFrame( sparse_matrix ) )


    def get_hist_of_top_followed( self, filepath ):
        '''
        Returns : count of friend followers
        --------------------------------------------------
        filepath : location to obtain data
        '''
        filepath = pd.DataFrame( filepath )

        plt.hist( filepath, bins = len( filepath ) )
        plt.title( 'Count of Friend Follower\'s' ).xlabel( 'Friend' ).ylabel( 'Count Following' )
        plt.show()
        

    def FindOptimalK( self ):
        return 

    
    def PerformKMeans( self ):
        return 

    # TODO // create features from descriptions