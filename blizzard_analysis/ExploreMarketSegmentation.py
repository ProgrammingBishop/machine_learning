import numpy             as np 
import pandas            as pd 
import matplotlib.pyplot as plt
import seaborn           as sns
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
        column_names                  = set( features )
        print(len( column_names) )
        print(len(features))
        # sparse_matrix                 = { 'ID' : follower_data[ 'screen_name' ] }
        # sparse_matrix[ column_names ] = 0

        # # For Every Follower
        # for follower in follower_data:
        #     # For Every Feature in Follower's Friends
        #     for feature in follower[ 'following' ]:
        #         # Flip Feature if Follower is Friends
        #         if feature in column_names:
        #             sparse_matrix[ feature ] = 1

        # self.save.write_to_csv_file( c.SPARSE_FRIENDS_MATRIX_CSV, pd.DataFrame( sparse_matrix ) )


    def get_barplot_of_top_followed( self, filepath, top_n ):
        '''
        Returns : barplot of blizzard followers most friended
        --------------------------------------------------
        filepath : location to obtain data
        top_n    : number of top most friended to plot (number + 1: upper-bound exclusive)
        '''
        sns.set( 
            style = "darkgrid",
            rc    = { 'figure.figsize' : ( 16, round( top_n / 5, 2 ) ) } 
        )

        top_followed_df     = pd.read_csv( filepath )
        top_followed_length = len( top_followed_df )
        top_followed_df     = top_followed_df[ 1:top_n ] # Skip 1st as it is Blizzard_Ent
        bar_values          = round( top_followed_df[ 'followed_by' ] / top_followed_length * 100 )
        g                   = sns.barplot( y = 'screen_name', x = 'followed_by', data = top_followed_df, color = "#00B4FF" )

        g.set_xlabel( 'Total Following', fontsize = 12 )
        g.tick_params( axis = 'x', labelsize = 10 )

        g.set_ylabel( 'Friend', fontsize = 12 )
        g.tick_params( axis = 'y', labelsize = 8 )

        g.set_title( 'Blizzard Followers most Friended', fontsize = 18 )

        for index, row in enumerate( bar_values ):
            g.text( 
                x        = top_followed_df.loc[ top_followed_df.index[ index ], 'followed_by' ] + 20,
                y        = index + 0.25,
                s        = str( row ) + '%', 
                color    = 'black',
                ha       = "center",
                fontsize = 8 
            )

        g.get_figure().savefig( c.FOLLOWERS_MOST_FRIENDED_PDF )
        

    def FindOptimalK( self ):
        return 

    
    def PerformKMeans( self ):
        return 

    # TODO // create features from descriptions