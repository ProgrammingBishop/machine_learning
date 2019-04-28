# Imports
# ==================================================
import configurations as c

from tweepy            import OAuthHandler
from DefineApplication import DefineApplication
from RunApplication    import RunApplication
        
# Entry Point
# ==================================================
if __name__ == '__main__':  
    authorize  = OAuthHandler( c.CONSUMER_KEY, c.CONSUMER_SECRET )
    authorize.set_access_token( c.ACCESS_TOKEN, c.ACCESS_TOKEN_SECRET )

    define_app = DefineApplication()
    steps      = define_app.define_program()

    RunApplication( steps, authorize, c )