from os.path import isfile

class Utilities():
    # PUBLIC
    def print_progress( self, progress, completion ):
        '''
        Return : prints progress of request
        --------------------------------------------------
        progress   : itreations to track
        completion : max iterations
        '''
        print( f"Progress: { str( round( progress / completion * 100, 2 ) ) }%" )


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