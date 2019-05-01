import spacy
import ast
import pandas         as pd
import configurations as c

from scipy import spatial
from Utilities       import Utilities

class GetDescriptors():
    # PRIVATE
    utilities   = Utilities()
    punctuation = '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n '
    __nlp = ''
    __str = ''
    
    def __init__( self ):
        self.__nlp = spacy.load( 'en_core_web_lg' )


    # PUBLIC
    def clean_text( self, description ):
        return_token = []
        token_lower  = ()

        for token in self.__nlp( description ):
            token_lower = token.text.lower()

            if token_lower not in self.punctuation:
                if token_lower not in self.__nlp.Defaults.stop_words:
                    if token_lower in self.__nlp.vocab:
                        if len( token_lower ) > 2:
                            return_token.append( token_lower )

        print( return_token )
        return return_token


    def get_descriptors( self ):
        # Create List of all IDs6
        data        = pd.read_csv( c.STREAM_DATAFRAME_CSV )
        # descriptors = data[ 'description' ].apply( lambda desc : ast.literal_eval( desc ) )
        unique_desc = []

        for index, desc in enumerate( data[ 'description' ] ):
            if index < 5:
                self.__str = self.__nlp( u'"' + str( desc ) + '"' )
                unique_desc += self.clean_text( str( desc ) )

        unique_desc = list( set( unique_desc ) )
        
        cosine_similarity       = lambda vec1, vec2: 1 - spatial.distance.cosine( vec1, vec2 )
        descriptor_similarities = {
            'word'        : [],
            'top_similar' : []
        }

        print( 'Next...\n' )

        for index, desc in enumerate( unique_desc ):
            if index < 5:
                computed_similarities = []
                a                     = self.__nlp.vocab[ str( desc ) ].vector

                for word in self.__nlp.vocab:
                    if word.has_vector:
                        if word.is_lower:
                            if word.is_alpha:
                                similarity = cosine_similarity( a, word.vector )
                                computed_similarities.append( ( word, similarity ) )

                computed_similarities = sorted( computed_similarities, key = lambda item : -item[ 1 ] )
                computed_similarities = [ t[ 0 ].text for t in computed_similarities[ :10 ] ]

                descriptor_similarities[ 'word'        ].append( desc )
                descriptor_similarities[ 'top_similar' ].append( computed_similarities )

                print( desc )
                print( computed_similarities )
                print( '\n' )

        self.utilities.write_to_file( c.TOKENIZED_DESCRIPTIONS, pd.DataFrame( descriptor_similarities ) )