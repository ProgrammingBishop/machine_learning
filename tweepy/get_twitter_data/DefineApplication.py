import sys

class DefineApplication:
    # PRIVATE
    def __get_user_input( self, message ):
        input_answer = ''

        while True:
            try:
                input_answer = str( input( message + " (t / f):" ) ).lower()
                print( '\n' )

                if input_answer == 't' or input_answer == 'f':
                    break
                else:
                    print( "Please answer with t of f" )
                    continue
            except:
                print( "Something went wrong..." )
                sys.exit()

        return input_answer


    # PUBLIC
    def define_program( self ):
        '''
        Return : list of bool values for steps to take
        --------------------------------------------------
        '''
        program_steps = []

        steps = [
            "Get related streamed tweet data?",
            "Get user's profile tweet data?",
            "Get user's follower data?",
            "Get friends of followers?"
        ]

        for step in steps:
            program_steps.append( self.__get_user_input( step ) )

        return program_steps