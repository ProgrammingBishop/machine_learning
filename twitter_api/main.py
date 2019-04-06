# Imports
# ==================================================
import configurations as c

from tweepy           import OAuthHandler
from Listener         import Listener, RobustStream
from GetFromTwitter   import GetFromTwitter


# Entry Point
# ==================================================
if __name__ == '__main__':  
    # Open Connection
    listener         = Listener()
    robust_stream    = RobustStream()
    get_from_twitter = GetFromTwitter()
    authorize        = OAuthHandler( c.CONSUMER_KEY, c.CONSUMER_SECRET )

    authorize.set_access_token( c.ACCESS_TOKEN, c.ACCESS_TOKEN_SECRET )

    # Stream Tweets on Tracked Topic/s
    # --------------------------------------------------
    robust_stream.start_stream( authorize, listener )
    get_from_twitter.stream_data_to_csv( c.STREAM_DATAFRAME_CSV )

    # Get Tweets from Tracked Profile
    # --------------------------------------------------
    get_from_twitter.status_data_to_csv( authorize, c.STATUSES_DATAFRAME_CSV )

    # Get Tracked Profile's Follower's Data
    # --------------------------------------------------
    get_from_twitter.get_follower_data( c.TARGET_SCREEN_NAME, authorize, c.FOLLOWER_DATA_CSV )

    # TODO
    # Get Tracked Profile's Follower's Friends
    # --------------------------------------------------