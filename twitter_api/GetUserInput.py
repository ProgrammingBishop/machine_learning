import sys

class GetUserInput:
    def __get_user_input( self, message ):
        input_answer = ''

        while True:
            try:
                input_answer = str( input( message + " (t / f):" ) ).lower()

                if input_answer == 't' or input_answer == 'f':
                    break
                else:
                    print( "Please answer with t of f" )
                    continue
            except:
                print( "Something went wrong..." )
                sys.exit()

        return input_answer


    def define_program( self ):
        program_steps = []

        program_steps.append( self.__get_user_input( "Get streamed tweet data?" ) )
        program_steps.append( self.__get_user_input( "Get profle's tweet data?" ) )
        program_steps.append( self.__get_user_input( "Get follower data?" ) )
        program_steps.append( self.__get_user_input( "Get friends of followers?" ) )

        return program_steps