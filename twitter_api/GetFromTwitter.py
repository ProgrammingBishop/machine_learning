from tweepy     import Cursor, API  
from pandas     import DataFrame, read_csv 
from SaveToFile import SaveToFile

import configurations as c
import time, sys, json, ast, collections

class GetFromTwitter():
    save = SaveToFile()

    def __get_tweets( self, tweet_file ):
        '''
        Return : JSON tweet data as Python list
        --------------------------------------------------
        tweet_file : Tweet stream text file to extract tweets from
        '''
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


    def stream_data_to_csv( self, filepath ):
        '''
        Return : CSV of Python list created by __get_tweets()
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
                if 'extended_tweet' in tweet:
                    extended_tweet = str( tweet[ 'extended_tweet' ] ).split( ": " )[1].split( "\', \'" )[0]
                    tweets_data[ 'text' ].append( extended_tweet )
                else:
                    tweets_data[ 'text' ].append( tweet[ 'text' ] )

                tweets_data[ 'screen_name' ].append( tweet[ 'user' ][ 'screen_name' ] )
                tweets_data[ 'description' ].append( tweet[ 'user' ][ 'description' ] )
                tweets_data[ 'created_at'  ].append( tweet[ 'user' ][ 'created_at'  ] )

        self.save.write_to_csv_file( filepath, DataFrame( tweets_data ) )


    def status_data_to_csv( self, authorize, filepath ):
        '''
        Return : CSV of api.user_timeline object
        created_at | full_text
        --------------------------------------------------
        authorize : tweepy OAuthHandler object
        filepath  : location to save output
        '''
        api = API( 
            authorize, 
            wait_on_rate_limit        = True,
            wait_on_rate_limit_notify = True
        )

        update_data = {
            'created_at' : [],
            'full_text'  : []
        }

        for update in Cursor( 
            api.user_timeline, 
            screen_name = c.TARGET_SCREEN_NAME, 
            lang        = 'en', 
            tweet_mode  = 'extended', 
            include_rts = True 
        ).items( c.END_STATUS_COUNT ):
            update_data[ 'created_at' ].append( update._json[ 'created_at' ] )
            update_data[ 'full_text'  ].append( update._json[ 'full_text' ] )

        self.save.write_to_csv_file( filepath, DataFrame( update_data ) )

    
    def get_follower_data( self, user_name, authorize, filepath ):
        '''
        Return : csv of followers and their friends
        screen_name | friends_ids
        --------------------------------------------------
        user_name : Twitter profile getting followers from
        authorize : tweepy OAuthHandler object
        filepath  : location to save output
        '''
        api = API( 
            authorize, 
            wait_on_rate_limit        = True,
            wait_on_rate_limit_notify = True
        )

        tracker       = 0
        follower_data = {
            'screen_name' : [],
            'description' : [],
            'location'    : []
        }

        for page in Cursor( api.followers, screen_name = user_name, languages = [ 'en' ] ).pages( c.MAX_FOLLOWER_PAGES ):
            print( "Progress: {}%"\
                .format( str( round( tracker / c.MAX_FOLLOWER_PAGES * 100, 2 ) ) ) )

            for follower in page:
                follower_data[ 'screen_name' ].append( follower._json[ 'screen_name' ] )
                follower_data[ 'description' ].append( follower._json[ 'description' ] )
                follower_data[ 'location'    ].append( follower._json[ 'location' ] )
            
            time.sleep( 60 )
            tracker += 1

        self.save.write_to_csv_file( filepath, DataFrame( follower_data ) )


    def get_follower_friends( self, authorize, filepath ):
        '''
        Return : CSV of target profile's follower's friends
        screen_name | following
        --------------------------------------------------
        authorize : tweepy OAuthHandler object
        filepath  : location to save output
        '''
        api = API( 
            authorize, 
            wait_on_rate_limit        = True,
            wait_on_rate_limit_notify = True
        )

        follower_friends = {
            'screen_name' : [],
            'following'   : []
        }

        try:
            follower_data = read_csv( c.FOLLOWER_DATA_CSV )
        except: 
            print( "File does not exist. Try running GetFromTwitter().get_follower_data()" )
        
        for index, follower in follower_data.iterrows():
            try: 
                follower_friends[ 'following'   ].append( str( api.friends_ids( follower[ 'screen_name' ] ) ) )
                follower_friends[ 'screen_name' ].append( follower[ 'screen_name' ] )
            except: 
                continue

            if index / 50 == 0:
                print( "Progress: {}%"\
                    .format( str( round( index / c.TOTAL_FOLLOWER_COUNT * 100, 2 ) ) ) )

        self.save.write_to_csv_file( filepath, DataFrame( follower_friends ) )


    def get_most_popular_friends( self, authorize, filepath ):
        '''
        Return : CSV of target profile's follower's friends
        screen_name | following
        --------------------------------------------------
        authorize : tweepy OAuthHandler object
        filepath  : location to save output
        '''
        api = API( 
            authorize, 
            wait_on_rate_limit        = True,
            wait_on_rate_limit_notify = True
        )

        # Create List of all IDs
        friends_data = read_csv( filepath )
        followers    = friends_data[ 'following' ].apply( lambda ids : ast.literal_eval( ids ) )

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
            current_friend = api.get_user( list( sorted_friend_counts.keys()   )[ friend ] )
            current_count  =               list( sorted_friend_counts.values() )[ friend ]

            top_friends_followed[ 'screen_name' ].append( current_friend.screen_name )
            top_friends_followed[ 'description' ].append( current_friend.description )
            top_friends_followed[ 'followed_by' ].append( current_count )

            if friend / 10 == 0:
                print( "Progress: {}%"\
                    .format( str( round( friend / c.TOP_MOST_FOLLOWED * 100, 2 ) ) ) )

        self.save.write_to_csv_file( c.TOP_FRIENDS_FOLLOWED_CSV, DataFrame( top_friends_followed ) )