DESKTOP = 'C:\\Users\\andre\\Desktop\\'

# File Locations 
# ==================================================
# Generated through get_from_twitter
STREAM_DATA_TXT          = DESKTOP + 'tweepy_output\\blizzard_stream_data.txt'
STREAM_DATAFRAME_CSV     = DESKTOP + 'blizzard_stream_dataframe.csv' # UPDATE AFTER PROTOTYPING
STREAM_DATAFRAME_CSV2     = DESKTOP + 'blizzard_stream_dataframe2.csv' # UPDATE AFTER PROTOTYPING

TARGET_STATUSES_TXT      = DESKTOP + 'tweepy_output\\blizzard_statuses_data.txt'
STATUSES_DATAFRAME_CSV   = DESKTOP + 'tweepy_output\\blizzard_statuses_dataframe.csv'

FOLLOWER_DATA_CSV        = DESKTOP + 'tweepy_output\\blizzard_follower_data.csv'
FOLLOWER_FRIENDS_CSV     = DESKTOP + 'tweepy_output\\blizzard_follower_friends.csv'
TOP_FRIENDS_FOLLOWED_CSV = DESKTOP + 'tweepy_output\\blizzard_top_friends_followed.csv'

# Generate through explore_twitter_data
FOLLOWERS_MOST_FRIENDED_PDF = DESKTOP + 'tweepy_output\\blizzard_followers_most_friended.pdf'
SPARSE_FRIENDS_MATRIX_CSV   = DESKTOP + 'tweepy_output\\blizzard_sparse_matrix.csv'
SPARSE_MATRIX_WLABELS_CSV   = DESKTOP + 'tweepy_output\\blizzard_sparse_matrix_wlabels.csv'
TOKENIZED_DESCRIPTIONS      = DESKTOP + 'tweepy_output\\blizzard_tokenized_descriptions.csv'
LABELED_DESCRIPTIONS        = DESKTOP + 'tweepy_output\\blizzard_labeled_descriptions.csv'


# Constants
# ==================================================
TOP_N    = 151
CLUSTERS = 0


# Notifications
# ==================================================
from twilio.rest import Client
from pandas      import read_csv

credentials = read_csv( DESKTOP + 'twilio_credentials.csv', delimiter = ',', index_col = None )
ACCOUNT_SID = credentials[ 'account_sid'  ][0]
AUTH_TOKEN  = credentials[ 'auth_token'   ][0]
PHONUE_NUM  = credentials[ 'phone_number' ][0]
client      = Client( ACCOUNT_SID, AUTH_TOKEN )