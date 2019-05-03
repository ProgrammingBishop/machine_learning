# Imports
# ==================================================
from sys import exit


# Class
# ==================================================
class DefineApplication:
    # PRIVATE
    def __get_user_input( self, message ):
        input_answer = ''

        while True:
            try:
                input_answer = str( input( message + " (t / f): " ) ).lower()
                print( '--------------------------------------------------\n' )

                if input_answer == 't' or input_answer == 'f':
                    break

                else:
                    print( "Please answer with t of f\n\n" )
                    continue

            except:
                print( "Something went wrong..." )
                exit()

        return input_answer


    # PUBLIC
    def define_program( self ):
        '''
        Return : list of bool values for steps to take
        --------------------------------------------------
        '''
        program_steps = []

        steps = [
            "Generate barplot of most followed users by target profile's followers?",
            "Cluster followers based on who they follow and how they describe themselves?"
        ]

        for step in steps:
            program_steps.append( self.__get_user_input( step ) )

        return program_steps