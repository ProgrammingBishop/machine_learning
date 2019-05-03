from os.path        import isfile
from sys            import exit
from configurations import client, PHONUE_NUM

class Utilities():
    # PUBLIC
    def print_progress( self, progress, completion ):
        '''
        Return : prints progress of request
        --------------------------------------------------
        progress   : itreations to track
        completion : max iterations
        '''
        print( f'Progress: { str( round( progress / completion * 100, 2 ) ) }%' )


    def write_to_file( self, filepath, content_to_write, write_mode = 'a', index = False ):
        '''
        Return : file containing data passed at location specified
        --------------------------------------------------
        filepath         : location to save output
        content_to_write : output to save at filepath 
        write_mode       : method to write to file (i.e "a" for append and "w" to clear and write)
        '''
        if '.csv' in filepath:
            content_to_write.to_csv( filepath, index = index )

        if '.txt' in filepath:
            with open( filepath, write_mode, encoding = 'utf8' ) as f:
                    f.write( content_to_write )

            
    def send_notification( self, message ):
        '''
        Return : sends text message regarding status of running application
        --------------------------------------------------
        message : SMS to send to phone number
        '''
        client.messages.create(
            to    = PHONUE_NUM, 
            from_ = '17085058854', 
            body  = message
        )

    def finding_file_error( self, missing_file, check_method ):
        '''
        Return : file not found error and suggestion on how to resolve error
        --------------------------------------------------
        missing_file : file method is searching for
        check_method : method to run that generates the file not found
        '''
        print( missing_file + ' in configurations is pointing to a non-existent file. Check path or generate file with the ' + check_method + ' method found in the get_from_twitter folder.' )
        exit()