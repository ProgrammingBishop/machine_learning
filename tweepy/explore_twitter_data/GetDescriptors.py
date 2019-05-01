import spacy
import ast
import pandas         as pd
import configurations as c

class GetDescriptors():
    # PRIVATE
    __nlp = ''
    __str = ''
    
    def __init__( self ):
        self.__nlp = spacy.load( 'en_core_web_lg' )


    # PUBLIC
    def clean_text( self, doc_text ):
        return [ token.text.lower() for token in self.__nlp( doc_text ) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ' and token.text not in self.__nlp.Defaults.stop_words ]


    def get_descriptors( self ):
        # Create List of all IDs
        data        = pd.read_csv( c.STREAM_DATAFRAME_CSV )
        # descriptors = data[ 'description' ].apply( lambda desc : ast.literal_eval( desc ) )
        unique_desc = []

        for desc in data[ 'description' ]:
            self.__str = self.__nlp( u'"' + str( desc ) + '"' )
            print( self.clean_text( str( desc ) ) )
            print( '\n' )
            
            # for token in self.__str:
            #     print( token.text )
            #     print( '\n' )
            # self.__str   = self.__nlp( u'"' + desc + '"' )
            # unique_desc.append( self.__str )
            # unique_desc += self.__str

        unique_desc = list( set( unique_desc ) )
        print( unique_desc )

        # Remove all stop words in this new list
        # Remove Punctuation

        # computed_similarities   = []
        # descriptor_similarities = {
            # word        = [],
            # top_similar = []
        # }

        # For desc in descriptions
            # desc = nlp.vocab[ desc ].vector

            # for word in nlp.vocab:
            #     if word.has_vector:
            #         if word.is_lower:
            #             if word.is_alpha:
            #                 similarity = cosine_similarity( desc, word.vector )
            #                 computed_similarities.append( ( word, similarity ) )

            # computed_similarities = sorted( computed_similarities, key = lambda item : -item[ 1 ] )
            # computed_similarities = [ t[ 0 ].text for t in computed_similarities[ :10 ] ]

            # descriptor_similarities[ 'word'        ].append( desc )
            # descriptor_similarities[ 'top_similar' ].append( desc )
        return