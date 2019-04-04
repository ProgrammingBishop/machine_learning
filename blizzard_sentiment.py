# Imports
# ==================================================
# Default Libraries
import sys

import numpy             as np 
import pandas            as pd 
import matplotlib.pyplot as plt 
import seaborn           as sns 

# Streaming Libraries
from tweepy.streaming import StreamListener
from tweepy           import OAuthHandler, Stream, Cursor, API

import facebook


# Global Variables, Classes, & Functions
# ==================================================
# Variables
TWITTER_WRITE_COUNT = 0
MAX_TWITTER_RECORDS = 1000

# Classes
class Listener( StreamListener ):
    def on_data( self, data ):
        global TWITTER_WRITE_COUNT
        global MAX_TWITTER_RECORDS

        # Write to File
        with open( '.\\blizzard_sentiment.txt', 'a', encoding = 'utf-8' ) as f:
            f.write( data )
            f.close()

        # Record Progress in Console
        TWITTER_WRITE_COUNT += 1

        print( 
            data + '\n' +
            str( TWITTER_WRITE_COUNT )
        )

        # End after N Status Records
        if TWITTER_WRITE_COUNT >= MAX_TWITTER_RECORDS:
            sys.exit()
        
    def on_error( self, status ):
        print( status )

# Functions


# Obtaining Twitter Data
# Find UserID: http://mytwitterid.com/
# ==================================================
# Securly Obtain Credentials
credentials = 'C: Users andre Desktop GitHub Data twitter_credentials twitter_credentials.csv'\
    .replace( " ", "\\" )

credentials = pd.read_csv( 
    credentials,
    delimiter = ',',
    index_col = None 
)

# Credentials
access_token        = credentials[ 'access_token'        ][0]
access_token_secret = credentials[ 'access_token_secret' ][0]
consumer_key        = credentials[ 'consumer_key'        ][0]
consumer_secret     = credentials[ 'consumer_secret'     ][0]

# Run Stream
if __name__ == '__main__':
    # Open Connection
    listener  = Listener()
    authorize = OAuthHandler( 
        consumer_key, 
        consumer_secret 
    )

    authorize.set_access_token( 
        access_token, 
        access_token_secret 
    )

    api             = API( authorize )

    # Build Stream Target
    user_stream     = Stream( auth = authorize, listener = listener )
    blizzard_stream = Stream( auth = authorize, listener = listener )

    # Initialize Stream
    try:
        # Obtain N Recent Statuses
        # blizzard = api.user_timeline( 
        #     screen_name = 'Blizzard_Ent', 
        #     count       = 100, 
        #     include_rts = True 
        # )

        # Stream Incoming Statuses
        user_stream.filter( 
            # Blizzard Entertainment
            follow   = '331908924',
            track    = [ 
                'Overwatch',   'Warcraft', 
                'Hearthstone', 'Diablo', 
                'Blizzard',    'StarCraft' 
            ], 
            encoding = 'utf8'
        )

    except KeyboardInterrupt:
        print( "Done" )

    finally:
        user_stream.disconnect()


# Obtaining Facebook Data
# ==================================================



# Obtaining YouTube Data 
# (Videos Targeted by Facebook & Twitter Trends)
# ==================================================