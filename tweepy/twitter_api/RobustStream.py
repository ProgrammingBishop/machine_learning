from tweepy.streaming import StreamListener
from tweepy           import Stream
from Utilities        import Utilities

import configurations  as c

class Listener( StreamListener ):
    # PRIVATE
    utility = Utilities()

    def on_data( self, data ):
        '''
        [Inherited tweepy Class]
        Return : txt file of Twitter stream data
        --------------------------------------------------
        data : Twitter information retreived by RobustStream() object
        '''
        self.utility.write_to_file( c.STREAM_DATA_TXT, data, 'a' )

        if c.START_STATUS_COUNT >= c.END_STATUS_COUNT:
            return False
        else:
            c.START_STATUS_COUNT += 1
            self.utility.print_progress( c.START_STATUS_COUNT, c.END_STATUS_COUNT )
        
    def on_error( self, status ):
        print( status )


class RobustStream():
    # PUBLIC
    def start_stream( self, authorize ):
        '''
        Return : txt file containing tweet information
        --------------------------------------------------
        authorize : tweepy OAuthHandler object
        listener  : tweepy Listener object (customized)
        '''
        listener = Listener()

        while ( c.START_STATUS_COUNT < c.END_STATUS_COUNT ):
            try:
                stream = Stream( auth = authorize, listener = listener, tweet_mode = "extended" )
                stream.filter( 
                    follow    = c.TARGET_TWITTER_ID,
                    track     = c.TARGET_TRACK_LIST, 
                    encoding  = 'utf8',
                    languages = [ 'en' ]
                )
            except:
                continue

        stream.disconnect()