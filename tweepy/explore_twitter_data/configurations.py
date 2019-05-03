# File Locations 
# ==================================================
# Generated through get_from_twitter
STREAM_DATA_TXT          = '.\\..\\..\\..\\tweepy_output\\blizzard_stream_data.txt'
TARGET_STATUSES_TXT      = '.\\..\\..\\..\\tweepy_output\\blizzard_statuses_data.txt'
STREAM_DATAFRAME_CSV     = '.\\..\\..\\..\\tweepy_output\\blizzard_stream_dataframe.csv'
STATUSES_DATAFRAME_CSV   = '.\\..\\..\\..\\tweepy_output\\blizzard_statuses_dataframe.csv'
FOLLOWER_DATA_CSV        = '.\\..\\..\\..\\tweepy_output\\blizzard_follower_data.csv'
FOLLOWER_FRIENDS_CSV     = '.\\..\\..\\..\\tweepy_output\\blizzard_follower_friends.csv'
TOP_FRIENDS_FOLLOWED_CSV = '.\\..\\..\\..\\tweepy_output\\blizzard_top_friends_followed.csv'

# Generate through explore_twitter_data
FOLLOWERS_MOST_FRIENDED_PDF = '.\\..\\..\\..\\plots\\blizzard_followers_most_friended.pdf'
SPARSE_FRIENDS_MATRIX_CSV   = '.\\..\\..\\..\\exploration_output\\blizzard_sparse_matrix.csv'
SPARSE_MATRIX_WLABELS_CSV   = '.\\..\\..\\..\\exploration_output\\blizzard_sparse_matrix_wlabels.csv'
TOKENIZED_DESCRIPTIONS      = '.\\..\\..\\..\\exploration_output\\blizzard_tokenized_descriptions.csv'
LABELED_DESCRIPTIONS        = '.\\..\\..\\..\\exploration_output\\blizzard_labeled_descriptions.csv'


# Constants
# ==================================================
TOP_N    = 151
CLUSTERS = 0


# Notifications
# ==================================================
from twilio.rest import Client
from pandas      import read_csv
import os

# Obtain Credentials
credentials = os.fspath( os.getcwd() )
move_up_dir = os.path.relpath( 'C:\\Users\\andre\\Desktop\\twilio_credentials', credentials )
credentials = os.path.join( credentials, move_up_dir + '\\twilio_credentials.csv' )
credentials = read_csv( credentials, delimiter = ',', index_col = None )

# Set Credentials
ACCOUNT_SID = credentials[ 'account_sid'  ][0]
AUTH_TOKEN  = credentials[ 'auth_token'   ][0]
PHONUE_NUM  = credentials[ 'phone_number' ][0]
client      = Client( ACCOUNT_SID, AUTH_TOKEN )

def send_notification( message ):
    client.messages.create(
        to    = PHONUE_NUM, 
        from_ = "17085058854", 
        body  = message
    )