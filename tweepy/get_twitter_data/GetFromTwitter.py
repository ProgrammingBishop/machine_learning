from tweepy     import Cursor, API  
from pandas     import DataFrame, read_csv
from Utilities  import Utilities

import configurations as c
import time, sys, json, ast, collections

class GetFromTwitter():
    # PRIVATE
    utility   = None
    authorize = None
    api       = None

    def __init__( self, authorize ):
        self.utility   = Utilities()
        self.authorize = authorize
        self.api       = API( 
            self.authorize, 
            wait_on_rate_limit        = True,
            wait_on_rate_limit_notify = True
        )


    def __get_tweets( self, tweet_file ):
        tweets = []

        try:
            tweets_file = open( tweet_file, "r" )
        except: 
            print( 'File open Error' )
            sys.exit()

        for line in tweets_file:
            try:
                tweet = json.loads( line )
                tweets.append( tweet )
            except:
                continue

        return tweets


    # PUBLIC
    def stream_to_csv( self, filepath ):
        '''
        Return : .csv containing the following features
        text | screen_name | description | created_at
        --------------------------------------------------
        filepath : location to save output
        '''
        tweets      = self.__get_tweets( c.STREAM_DATA_TXT )
        tweets_data = {
            'text'          : [],
            'screen_name'   : [],
            'description'   : [],
            'created_at'    : []
        }

        for tweet in tweets:
            if ( not tweet[ 'retweeted' ] ) and ( 'RT @' not in tweet[ 'text' ] ):
                tweet_text = tweet[ 'text' ]

                if 'extended_tweet' in tweet:
                    tweet_text = str( tweet[ 'extended_tweet' ] ).split( ": " )[1].split( "\', \'" )[0]

                tweets_data[ 'text' ].append( tweet_text )

                for feature in tweets_data.keys():
                    if feature == 'text':
                        continue

                    tweets_data[ feature ].append( tweet[ 'user' ][ feature ] )

        self.utility.write_to_file( filepath, DataFrame( tweets_data ) )


    def tweets_to_csv( self, filepath ):
        '''
        Return : .csv containing the following features
        created_at | full_text
        --------------------------------------------------
        filepath  : location to save output
        '''
        update_data = {
            'created_at' : [],
            'full_text'  : []
        }

        for update in Cursor( 
            self.api.user_timeline, 
            screen_name = c.TARGET_SCREEN_NAME, 
            lang        = 'en', 
            tweet_mode  = 'extended', 
            include_rts = True 
        ).items( c.END_STATUS_COUNT ):
            for feature in update_data.keys():
                update_data[ feature ].append( update._json[ feature ] )

        self.utility.write_to_file( filepath, DataFrame( update_data ) )

    
    def followers_to_csv( self, user_name, filepath ):
        '''
        Return : .csv containing the following features
        screen_name | friend_ids
        --------------------------------------------------
        user_name : Twitter profile getting followers from
        filepath  : location to save output
        '''
        follower_data = {
            'screen_name' : [],
            'description' : [],
            'location'    : []
        }

        for tracker, page in enumerate( 
            Cursor( 
                self.api.followers, 
                screen_name = user_name, 
                languages   = [ 'en' ] 
            ).pages( c.MAX_FOLLOWER_PAGES ) 
        ):
            self.utility.print_progress( tracker, c.MAX_FOLLOWER_PAGES )

            try:
                for follower in page:
                    for feature in follower_data.keys():
                        follower_data[ feature ].append( follower._json[ feature ] )
                
                time.sleep( 60 )
            except:
                continue

        self.utility.write_to_file( filepath, DataFrame( follower_data ) )


    def follower_friends_to_csv( self, filepath ):
        '''
        Return : .csv containing the following features
        screen_name | following
        --------------------------------------------------
        filepath  : location to save output
        '''
        follower_friends = {
            'screen_name' : [],
            'following'   : []
        }

        try:
            follower_data = read_csv( c.FOLLOWER_DATA_CSV )
        except: 
            print( "File does not exist. Try running GetFromTwitter().followers_to_csv()" )
        
        for index, follower in follower_data.iterrows():
            try: 
                follower_friends[ 'following'   ].append( str( self.api.friends_ids( follower[ 'screen_name' ] ) ) )
                follower_friends[ 'screen_name' ].append( follower[ 'screen_name' ] )
            except: 
                continue

            if index % 5 == 0:
                self.utility.print_progress( index, c.FOLLOWER_ITERATONS )

            if index == c.FOLLOWER_ITERATONS:
                break

        self.utility.write_to_file( filepath, DataFrame( follower_friends ) )


    def popular_friends_to_csv( self, filepath ):
        '''
        Return : .csv containing the following features
        screen_name | following
        --------------------------------------------------
        filepath  : location to save output
        '''
        # Create List of all IDs
        friends_data     = read_csv( filepath )
        followers        = friends_data[ 'following' ].apply( lambda ids : ast.literal_eval( ids ) )
        followed_friends = []

        for friends in followers:
            followed_friends += friends

        # Get Count of each IDs Occurrence
        friend_counts        = collections.Counter( followed_friends )
        sorted_friend_counts = collections.OrderedDict( sorted( 
            friend_counts.items(), 
            key = lambda kv: 
            kv[1], 
            reverse = True
        ) )

        # Get DataFrame of Top Friended
        top_friends_followed = {
            'screen_name' : [],
            'description' : [],
            'followed_by' : []
        }

        current_friend = None

        for friend in range( c.TOP_MOST_FOLLOWED ):
            try:
                current_friend = self.api.get_user( list( sorted_friend_counts.keys()   )[ friend ] )
                current_count  =                    list( sorted_friend_counts.values() )[ friend ]

                top_friends_followed[ 'screen_name' ].append( current_friend.screen_name )
                top_friends_followed[ 'description' ].append( current_friend.description )
                top_friends_followed[ 'followed_by' ].append( current_count )
            except: 
                continue

            if friend % 10 == 0:
                self.utility.print_progress( friend, c.TOP_MOST_FOLLOWED )

        self.utility.write_to_file( c.TOP_FRIENDS_FOLLOWED_CSV, DataFrame( top_friends_followed ) )