class SaveToFile():
    def write_to_text_file( self, filepath, content_to_write, write_mode ):
        '''
        filepath         : location to save output
        content_to_write : content to save at filepath argument 
        write_mode       : mwthod to write to file (i.e "a" for append and "w" to clear and write)
        '''
        with open( filepath, write_mode, encoding = 'utf8' ) as f:
                f.write( content_to_write )
                f.close()

    def write_to_csv_file( self, filepath, content_to_write, index = False ):
        '''
        filepath         : location to save output
        content_to_write : content to save at filepath argument 
        '''
        content_to_write.to_csv( filepath, index = index )