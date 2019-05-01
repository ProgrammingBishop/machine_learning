import os, pandas, math

# Obtain Credentials
credentials = os.fspath( os.getcwd() )
move_up_dir = os.path.relpath( 'C:\\Users\\andre\\Desktop\\twitter_credentials', credentials )
credentials = os.path.join( credentials, move_up_dir + '\\twitter_credentials.csv' )
credentials = pandas.read_csv( credentials, delimiter = ',', index_col = None )

# Set Credentials
ACCESS_TOKEN        = credentials[ 'access_token'        ][0]
ACCESS_TOKEN_SECRET = credentials[ 'access_token_secret' ][0]
CONSUMER_KEY        = credentials[ 'consumer_key'        ][0]
CONSUMER_SECRET     = credentials[ 'consumer_secret'     ][0]

# Target Twitter User
# http://gettwitterid.com/
TARGET_TWITTER_ID  = '331908924'
TARGET_SCREEN_NAME = 'Blizzard_Ent'
TARGET_TRACK_LIST  = [ 
    'Overwatch',   'Warcraft',  'Hearthstone', 'Diablo', 
    'Blizzard',    'StarCraft', 'Esports',     'BlizzCon' 
]

# Stream Quantities
START_STATUS_COUNT   = 0
END_STATUS_COUNT     = 6000
TOTAL_FOLLOWER_COUNT = 5000
MAX_FOLLOWER_PAGES   = math.ceil( TOTAL_FOLLOWER_COUNT / 20 ) # Pages = Total / Number per Page
# Follower Iterations Breakdown:
    # Refreshes = Followers Tracked / 15 (Rate Limit Hit every 15 Requests)
    # Minutes   = Refreshes * 15 (Refresh Time for Next Request is 15 Minutes)
    # Hours     = Minutes / 60
    # Example   : 750 Followers Tracked = 12.5 Hours to Complete
FOLLOWER_ITERATONS = 5000 
TOP_MOST_FOLLOWED  = 150

# Save File Locations
STREAM_DATA_TXT          = '.\\..\\..\\..\\tweepy_output\\blizzard_stream_data.txt'
TARGET_STATUSES_TXT      = '.\\..\\..\\..\\tweepy_output\\blizzard_statuses_data.txt'
STREAM_DATAFRAME_CSV     = '.\\..\\..\\..\\tweepy_output\\blizzard_stream_dataframe.csv'
STATUSES_DATAFRAME_CSV   = '.\\..\\..\\..\\tweepy_output\\blizzard_statuses_dataframe.csv'
FOLLOWER_DATA_CSV        = '.\\..\\..\\..\\tweepy_output\\blizzard_follower_data.csv'
FOLLOWER_FRIENDS_CSV     = '.\\..\\..\\..\\tweepy_output\\blizzard_follower_friends.csv'
TOP_FRIENDS_FOLLOWED_CSV = '.\\..\\..\\..\\tweepy_output\\blizzard_top_friends_followed.csv'