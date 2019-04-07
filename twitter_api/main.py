# Imports
# ==================================================
import configurations as c

from tweepy           import OAuthHandler
from Listener         import Listener, RobustStream
from GetFromTwitter   import GetFromTwitter
from GetUserInput     import GetUserInput
             
        
# Entry Point
# ==================================================
if __name__ == '__main__':  
    # Instantiate Classes
    get_user_input   = GetUserInput()
    listener         = Listener()
    robust_stream    = RobustStream()
    get_from_twitter = GetFromTwitter()

    # Open Connection
    authorize  = OAuthHandler( c.CONSUMER_KEY, c.CONSUMER_SECRET )
    authorize.set_access_token( c.ACCESS_TOKEN, c.ACCESS_TOKEN_SECRET )

    # Define Steps
    steps = get_user_input.define_program()

    # Stream Tweets on Tracked Topic/s
    # --------------------------------------------------
    if steps[0] == 't':
        print( "Getting streamed tweet data..." )
        robust_stream.start_stream( authorize, listener )
        get_from_twitter.stream_data_to_csv( c.STREAM_DATAFRAME_CSV )

    # Get Tweets from Tracked Profile
    # --------------------------------------------------
    if steps[1] == 't':
        print( "Getting target profile's tweet data..." )
        get_from_twitter.status_data_to_csv( authorize, c.STATUSES_DATAFRAME_CSV )

    # Get Tracked Profile's Follower's Data
    # --------------------------------------------------
    if steps[2] == 't':
        print( "Getting target profile's follower data..." )
        get_from_twitter.get_follower_data( c.TARGET_SCREEN_NAME, authorize, c.FOLLOWER_DATA_CSV )

        if steps[3] == 't':
            print( "Now getting the follower's friends..." )
            # Get Tracked Profile's Follower's Friends
            # --------------------------------------------------
            get_from_twitter.get_follower_friends( authorize, c.FOLLOWER_FRIENDS_CSV )