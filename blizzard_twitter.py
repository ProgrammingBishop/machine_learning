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
import ast

# Streaming Libraries
from tweepy.streaming import StreamListener
from tweepy           import OAuthHandler, Stream, Cursor, API, TweepError


# Global Variables
# ==================================================
# Find UserID: http://mytwitterid.com/
TARGET_TWITTER_ID  = '331908924'
TARGET_SCREEN_NAME = 'Blizzard_Ent'
TARGET_TRACK_LIST  = [ 
    'Overwatch',   'Warcraft', 
    'Hearthstone', 'Diablo', 
    'Blizzard',    'StarCraft',
    'Esports' 
]
START_STATUS_COUNT     = 0
END_STATUS_COUNT       = 5000
MAX_FOLLOWERS_RETURNED = 500

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
STREAM_DATA_TXT        = '.\\..\\..\\output\\blizzard_stream_data.txt'
TARGET_STATUSES_TXT    = '.\\..\\..\\output\\blizzard_statuses_data.txt'
STREAM_DATAFRAME_CSV   = '.\\..\\..\\output\\blizzard_stream_dataframe.csv'
STATUSES_DATAFRAME_CSV = '.\\..\\..\\output\\blizzard_statuses_dataframe.csv'
FOLLOWER_FRIENDS_CSV   = '.\\..\\..\\output\\blizzard_follower_friends.csv'


# Classes
# ==================================================
class SaveToFile():
    def write_to_text_file( self, filepath, content_to_write, write_mode ):
        '''
        filepath         : location to save file
        content_to_write : content to save at filepath specified 
        write_mode       : mwthod to write to file (i.e "a" for append and "w" to clear and write)
        '''
        with open( filepath, write_mode, encoding = 'utf8' ) as f:
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
        [Extended tweepy Class]
        Return : txt file of Twitter stream data
        --------------------------------------------------
        data : Twitter information retreived by Stream() object
        '''
        global START_STATUS_COUNT, END_STATUS_COUNT, STREAM_DATA_TXT

        save = SaveToFile()
        save.write_to_text_file( STREAM_DATA_TXT, data, 'a' )

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
        Return : csv of followers and their friends
        screen_name | friends_ids
        --------------------------------------------------
        user_name : Twitter profile getting followers from
        api       : teepy API() object3
        '''
        global MAX_FOLLOWERS_RETURNED, FOLLOWER_FRIENDS_CSV

        save             = SaveToFile()
        follower         = ''
        follower_friends = {}

        for follower in Cursor( api.followers, screen_name = user_name, lang = 'en' ).items( MAX_FOLLOWERS_RETURNED ):
            follower_name                     = follower._json[ 'screen_name' ]
            friend_ids                        = str( api.friends_ids( screen_name = follower_name ) )
            follower_friends[ follower_name ] = friend_ids
        
        save.write_to_csv_file( FOLLOWER_FRIENDS_CSV, pd.DataFrame( follower_friends, index = [ 'friends_ids' ] ) )


    def get_tweets( self, tweet_file ):
        '''
        Return : JSON tweet data as Python list
        --------------------------------------------------
        tweet_file : text file to extract tweets from
        '''
        tweets = []

        # Read from File
        try:
            tweets_file = open( tweet_file, "r" )
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


    def stream_data_to_csv( self, tweets, filepath ):
        '''
        Return : CSV of Python list created by get_tweets()
        text | screen_name | description | created_at
        --------------------------------------------------
        tweets   : Python list
        filepath : path to file to save data to
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
                if 'extended_tweet' in tweet:
                    extended_tweet = str( tweet[ 'extended_tweet' ] ).split( ": " )[1].split( "\', \'" )[0]
                    tweets_data[ 'text' ].append( extended_tweet )
                else:
                    tweets_data[ 'text' ].append( tweet[ 'text' ] )

                tweets_data[ 'screen_name' ].append( tweet[ 'user' ][ 'screen_name' ] )
                tweets_data[ 'description' ].append( tweet[ 'user' ][ 'description' ] )
                tweets_data[ 'created_at'  ].append( tweet[ 'user' ][ 'created_at'  ] )

        save.write_to_csv_file( filepath, pd.DataFrame( tweets_data ) )


    def status_data_to_csv( self, api, filepath ):
        '''
        Return : CSV of api.user_timeline object
        created_at | full_text
        --------------------------------------------------
        api      : tweepy api object
        filepath : path to file to save data to
        '''
        global TARGET_SCREEN_NAME, END_STATUS_COUNT

        save        = SaveToFile()
        update_data = {
            'created_at' : [],
            'full_text'  : []
        }

        for update in Cursor( 
            api.user_timeline, 
            screen_name = TARGET_SCREEN_NAME, 
            lang        = 'en', 
            tweet_mode  = 'extended'
        ).items( END_STATUS_COUNT ):
            update_data[ 'created_at' ].append( update._json[ 'created_at' ] )
            update_data[ 'full_text'  ].append( update._json[ 'full_text' ] )

            print( "Progress: {}%"\
                .format( str( round( len( update_data[ 'full_text' ] ) / END_STATUS_COUNT * 100, 2 ) ) ) )

        save.write_to_csv_file( filepath, pd.DataFrame( update_data ) )


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

    # Stream 1 : Get Incoming Tweets on Tracked Topic
    # --------------------------------------------------
    # [Custom Function]: Restart Stream if Read Timed Out
    # def start_stream():
    #     while ( START_STATUS_COUNT < END_STATUS_COUNT ):
    #         try:
    #             stream = Stream( auth = authorize, listener = listener, tweet_mode = "extended" )
    #             stream.filter( 
    #                 follow    = TARGET_TWITTER_ID,
    #                 track     = TARGET_TRACK_LIST, 
    #                 encoding  = 'utf8',
    #                 languages = [ 'en' ]
    #             )
    #         except: 
    #             continue
            
    #     stream.disconnect()

    # start_stream()

    # tweets = get_from_twitter.get_tweets( STREAM_DATA_TXT )
    # get_from_twitter.stream_data_to_csv( tweets, STREAM_DATAFRAME_CSV )


    # Stream 2 : Get Tweets from Tracked Profile
    # --------------------------------------------------
    api = API( 
        authorize, 
        wait_on_rate_limit        = True,
        wait_on_rate_limit_notify = True
    )

    get_from_twitter.status_data_to_csv( api, STATUSES_DATAFRAME_CSV )


    # Stream 3 : Get Tracked Profile's Follower's Friends
    # --------------------------------------------------
    # get_from_twitter.get_follower_friends( TARGET_SCREEN_NAME, api )


# data = pd.read_csv( '.\\..\\..\\output\\blizzard_follower_friends.csv' ).transpose()
# ast.literal_eval( dict( data )[0][0] )