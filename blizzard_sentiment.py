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
START_STATUS_COUNT   = 0
END_STATUS_COUNT     = 1000
MAX_FOLLOWERS_REPORT = 2

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
TARGET_STREAM_DATA   = '.\\blizzard_stream_data.txt'
TARGET_STATUSES_DATA = '.\\blizzard_statuses_data.txt'
FOLLOWER_FRIEND_DATA = '.\\blizzard_follower_friends.txt'


# Functions
# ==================================================
def write_to_file( filepath, content_to_write ):
    with open( filepath, 'a', encoding = 'utf-8' ) as f:
            f.write( content_to_write ).close()


# Classes
# ==================================================
class Listener( StreamListener ):
    def on_data( self, data ):
        '''
        Saves Twitter stream data into text file
        --------------------------------------------------
        data     : Twitter information retreived by Stream() object
        filename : Location to write data to
        '''
        global START_STATUS_COUNT
        global END_STATUS_COUNT
        global TARGET_STREAM_DATA

        write_to_file( TARGET_STREAM_DATA, data )

        if START_STATUS_COUNT >= END_STATUS_COUNT:
            sys.exit()
        else:
            START_STATUS_COUNT += 1
            print( "Progress: {}%"\
                .format( str( round( START_STATUS_COUNT / END_STATUS_COUNT * 100, 2 ) ) ) )
        
    def on_error( self, status ):
        print( status )


class GetFromTwitter():
    def get_follower_friends( self, user_name, api ):
        '''
        Saves dictionary of user's follower's friends.
        --------------------------------------------------
        user_name : Twitter profile getting followers from
        api       : teepy API() object
        '''
        global MAX_FOLLOWERS_REPORT
        global FOLLOWER_FRIEND_DATA

        follower_friends = defaultdict( list )

        for follower in Cursor( api.followers, screen_name = user_name, ).items( MAX_FOLLOWERS_REPORT ):
            follower_friends[ '"' + follower.screen_name +  '"' ]\
                .append( api.friends_ids( screen_name = follower.screen_name ) )

        write_to_file( FOLLOWER_FRIEND_DATA, follower_friends )


# Obtaining Data
# ==================================================
# Run Stream
if __name__ == '__main__':   
    # Open Connection
    get_from_twitter = GetFromTwitter()
    listener         = Listener()
    authorize        = OAuthHandler( CONSUMER_KEY, CONSUMER_SECRET )

    authorize.set_access_token( ACCESS_TOKEN, ACCESS_TOKEN_SECRET )

    # Build Stream
    api    = API( authorize, wait_on_rate_limit = True )
    stream = Stream( auth = authorize, listener = listener )

    # Retrieve/Write Statuses
    target_statuses = api.user_timeline( 
        screen_name = TARGET_SCREEN_NAME, 
        count       = END_STATUS_COUNT, 
        include_rts = True 
    )

    write_to_file( TARGET_STATUSES_DATA, target_statuses )

    # Retrieve/Write Follower Friends
    get_from_twitter.get_follower_friends( TARGET_SCREEN_NAME, api )

    # Retrieve/Write Streamed Statuses
    stream.filter( 
        follow   = TARGET_TWITTER_ID,
        track    = TARGET_TRACK_LIST, 
        encoding = 'utf8'
    )