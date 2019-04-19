from tweepy.streaming import StreamListener
from tweepy           import Stream
from SaveToFile       import SaveToFile

import configurations  as c

class Listener( StreamListener ):
    save = SaveToFile()

    def on_data( self, data ):
        '''
        [Inherited tweepy Class]
        Return : txt file of Twitter stream data
        --------------------------------------------------
        data : Twitter information retreived by RobustStream() object
        '''
        self.save.write_to_text_file( c.STREAM_DATA_TXT, data, 'a' )

        if c.START_STATUS_COUNT >= c.END_STATUS_COUNT:
            return False
        else:
            c.START_STATUS_COUNT += 1
            print( "Progress: {}%"\
                .format( str( round( c.START_STATUS_COUNT / c.END_STATUS_COUNT * 100, 2 ) ) ) )
        
    def on_error( self, status ):
        print( status )


class RobustStream():
    def start_stream( self, authorize, listener ):
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