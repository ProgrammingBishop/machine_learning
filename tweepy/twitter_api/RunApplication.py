from RobustStream      import RobustStream
from GetFromTwitter    import GetFromTwitter

class RunApplication():
    def __init__( self, steps, authorize, c ):
        robust_stream = RobustStream()
        twitter       = GetFromTwitter( authorize )

        # Stream Tweets for Target User
        # --------------------------------------------------
        if steps[0] == 't':
            print( "Getting relevant streamed tweet data..." )
            robust_stream.start_stream( authorize )
            twitter.stream_to_csv( c.STREAM_DATAFRAME_CSV )

        # Get Tweets from Tracked Profile
        # --------------------------------------------------
        if steps[1] == 't':
            print( "Getting target user's tweet data..." )
            twitter.tweets_to_csv( c.STATUSES_DATAFRAME_CSV )

        # # Get Tracked Profile's Follower's Data
        # # --------------------------------------------------
        if steps[2] == 't':
            print( "Getting target user's follower data..." )
            twitter.followers_to_csv( c.TARGET_SCREEN_NAME, c.FOLLOWER_DATA_CSV )

            if steps[3] == 't':
                print( "Now getting the follower's friends..." )
                twitter.follower_friends_to_csv( c.FOLLOWER_FRIENDS_CSV )

                print( "Lastly getting the top-most followed friends' data..." )
                twitter.popular_friends_to_csv( c.FOLLOWER_FRIENDS_CSV )