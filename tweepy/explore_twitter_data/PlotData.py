import matplotlib.pyplot as plt 
import seaborn           as sns 
import pandas            as pd 


class PlotData():
    # PUBLIC
    def get_barplot_pdf( self, c ):
        '''
        Returns : pdf of barplot of most followed users by target profile's followers
        --------------------------------------------------
        '''
        sns.set( 
            style = "darkgrid",
            rc    = { 'figure.figsize' : ( 16, round( c.TOP_N / 5, 2 ) ) } 
        )

        top_followed_df     = pd.read_csv( c.TOP_FRIENDS_FOLLOWED_CSV )
        top_followed_length = len( top_followed_df )
        top_followed_df     = top_followed_df[ 1:c.TOP_N ] # Skip 1st as it is Blizzard_Ent
        bar_values          = round( top_followed_df[ 'followed_by' ] / top_followed_length * 100 )
        g                   = sns.barplot( y = 'screen_name', x = 'followed_by', data = top_followed_df, color = "#00B4FF" )

        g.set_xlabel( 'Total Following', fontsize = 12 )
        g.tick_params( axis = 'x', labelsize = 10 )

        g.set_ylabel( 'Friend', fontsize = 12 )
        g.tick_params( axis = 'y', labelsize = 8 )

        g.set_title( 'Blizzard Followers most Friended', fontsize = 18 )

        # Add Percentage for Each Bar
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