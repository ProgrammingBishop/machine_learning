# Imports
# ==================================================
# Default Libraries
import numpy             as np 
import pandas            as pd 
import matplotlib.pyplot as plt 
import seaborn           as sns 

from collections import defaultdict

# System Libraries
import sys
import time
import os
import re
import json

# Streaming Libraries
from tweepy.streaming import StreamListener
from tweepy           import OAuthHandler, Stream, Cursor, API, TweepError


# Global Variables
# ==================================================
# Find UserID: http://mytwitterid.com/
TARGET_SCREEN_NAME       = 'Blizzard_Ent'
TARGET_TWITTER_ID        = '331908924'
TARGET_TRACK_LIST        = [ 
    'Overwatch',   'Warcraft', 
    'Hearthstone', 'Diablo', 
    'Blizzard',    'StarCraft' 
]
START_STATUS_COUNT    = 1
END_STATUS_COUNT      = 15
MAX_FOLLOWERS_TRACKED = 2

# Securly Obtain Credentials
credentials = os.fspath( os.getcwd() )
move_up_dir = os.path.relpath( 'C:\\Users\\andre\\Desktop\\twitter_credentials', credentials )
credentials = os.path.join( credentials, move_up_dir + '\\twitter_credentials.csv' )
credentials = pd.read_csv( credentials, delimiter = ',', index_col = None )

# Credentials
ACCESS_TOKEN        = credentials[ 'access_token'        ][0]
ACCESS_TOKEN_SECRET = credentials[ 'access_token_secret' ][0]
CONSUMER_KEY        = credentials[ 'consumer_key'        ][0]
CONSUMER_SECRET     = credentials[ 'consumer_secret'     ][0]

# Save File Locations
TARGET_STREAM_DATA   = '.\\..\\..\\output\\blizzard_stream_data.txt'
STREAM_DATA_FRAME    = '.\\..\\..\\output\\blizzard_stream_data_frame.csv'
TARGET_STATUSES_DATA = '.\\..\\..\\output\\blizzard_statuses_data.txt'
FOLLOWER_FRIEND_DATA = '.\\..\\..\\output\\blizzard_follower_friends.txt'


# Classes
# ==================================================
class SaveToFile():
    def write_to_text_file( self, filepath, content_to_write ):
        '''
        filepath         : location to save file
        content_to_write : content to save at filepath specified 
        '''
        with open( filepath, 'a', encoding = 'utf8' ) as f:
                f.write( content_to_write )
                f.close()

    def write_to_csv_file( self, filepath, content_to_write ):
        '''
        filepath         : location to save file
        content_to_write : content to save at filepath specified 
        '''
        content_to_write.to_csv( filepath, index = False )


class Listener( StreamListener ):
    def on_data( self, data ):
        '''
        [Extended tweepy Class]: Save Twitter stream data into text file
        --------------------------------------------------
        data     : Twitter information retreived by Stream() object
        filename : Location to write data to
        '''
        global START_STATUS_COUNT
        global END_STATUS_COUNT
        global TARGET_STREAM_DATA
        global SAVE_TO_FILE

        save = SaveToFile()

        save.write_to_text_file( TARGET_STREAM_DATA, data )

        if START_STATUS_COUNT >= END_STATUS_COUNT:
            return False
        else:
            START_STATUS_COUNT += 1
            print( "Progress: {}%"\
                .format( str( round( START_STATUS_COUNT / END_STATUS_COUNT * 100, 2 ) ) ) )
        
    def on_error( self, status ):
        print( status )


class GetFromTwitter():
    def get_follower_friends( self, user_name, api ):
        '''
        Save dictionary of user's follower's friends in text file
        --------------------------------------------------
        user_name : Twitter profile getting followers from
        api       : teepy API() object
        '''
        global MAX_FOLLOWERS_TRACKED
        global FOLLOWER_FRIEND_DATA
        global SAVE_TO_FILE

        save             = SaveToFile()
        follower_friends = defaultdict( list )

        for follower in Cursor( api.followers, screen_name = user_name, lang='en' ).items( MAX_FOLLOWERS_TRACKED ):
            follower_friends[ '"' + follower.screen_name +  '"' ]\
                .append( api.friends_ids( screen_name = follower.screen_name ) )

        save.write_to_text_file( FOLLOWER_FRIEND_DATA, follower_friends )


    def get_tweets( self ):
        '''
        Convert JSON Tweet data into Python list object
        '''
        global TARGET_STREAM_DATA
        global SAVE_TO_FILE

        tweets = []

        # Read from File
        try:
            tweets_file = open( TARGET_STREAM_DATA, "r" )
        except: 
            print( 'File open Error' )
            sys.exit()

        # Convert User Object into JSON Object
        for line in tweets_file:
            try:
                tweet = json.loads( line )
                tweets.append( tweet )
            except:
                continue

        return tweets


    def tweet_data_to_csv( self, tweets ):
        '''
        Convert Python list object into CSV Dataframe
        text | screen_name | description | created_at
        --------------------------------------------------
        tweets : JSON object created from tweepy User object
        '''
        save        = SaveToFile()
        tweets_data = {
            'text'          : [],
            'screen_name'   : [],
            'description'   : [],
            'created_at'    : []
        }

        for tweet in tweets:
            # Ignore Retweets
            if ( not tweet[ 'retweeted' ] ) and ( 'RT @' not in tweet[ 'text' ] ):
                # Handle Longer Tweets
                if 'extended_tweet' in tweet:
                    extended_tweet = str( tweet[ 'extended_tweet' ] ).split( ": " )[1].split( "\', \'" )[0]
                    print( extended_tweet )
                    print( '\n' )
                    tweets_data[ 'text' ].append( extended_tweet )
                else:
                    tweets_data[ 'text' ].append( tweet[ 'text' ] )

                tweets_data[ 'screen_name' ].append( tweet[ 'user' ][ 'screen_name' ] )
                tweets_data[ 'description' ].append( tweet[ 'user' ][ 'description' ] )
                tweets_data[ 'created_at'  ].append( tweet[ 'user' ][ 'created_at'  ] )

        save.write_to_csv_file( STREAM_DATA_FRAME, pd.DataFrame( tweets_data ) )


# Obtaining Data
# ==================================================
# Run Stream
if __name__ == '__main__':  
    save = SaveToFile()

    # Open Connection
    get_from_twitter = GetFromTwitter()
    listener         = Listener()
    authorize        = OAuthHandler( CONSUMER_KEY, CONSUMER_SECRET )

    authorize.set_access_token( ACCESS_TOKEN, ACCESS_TOKEN_SECRET )

    # Build Stream
    api    = API( authorize, wait_on_rate_limit = True )
    stream = Stream( auth = authorize, listener = listener, tweet_mode = "extended" )

    # Retrieve/Write User's Statuses
    # target_statuses = api.user_timeline( 
    #     screen_name = TARGET_SCREEN_NAME, 
    #     count       = END_STATUS_COUNT, 
    #     include_rts = True 
    # )

    # save.write_to_text_file( TARGET_STATUSES_DATA, target_statuses )

    # Retrieve/Write Follower Friends
    # get_from_twitter.get_follower_friends( TARGET_SCREEN_NAME, api )

    # Retrieve/Write Streamed Statuses
    stream.filter( 
        follow    = TARGET_TWITTER_ID,
        track     = TARGET_TRACK_LIST, 
        encoding  = 'utf8',
        languages = [ 'en' ]
    )

    tweets = get_from_twitter.get_tweets()
    get_from_twitter.tweet_data_to_csv( tweets )