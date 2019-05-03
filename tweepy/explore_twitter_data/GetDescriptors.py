# Imports
# ==================================================
from scipy                           import spatial
from Utilities                       import Utilities
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition           import LatentDirichletAllocation

import spacy
import ast
import pandas         as pd
import configurations as c


# Class
# ==================================================
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
            if index < 50:
                self.__str = self.__nlp( u'"' + str( desc ) + '"' )
                unique_desc += self.clean_text( str( desc ) )

        unique_desc = list( set( unique_desc ) )
        
        cosine_similarity       = lambda vec1, vec2: 1 - spatial.distance.cosine( vec1, vec2 )
        descriptor_similarities = {
            'word'        : [],
            'top_similar' : []
        }

        for index, desc in enumerate( unique_desc ):
            if index < 50:
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

        self.utilities.write_to_file( c.TOKENIZED_DESCRIPTIONS, pd.DataFrame( descriptor_similarities ) )

    
    def topic_model_descriptors( self ):
        df  = pd.read_csv( c.TOKENIZED_DESCRIPTIONS )
        cv  = CountVectorizer( stop_words = 'english' )
        dtm = cv.fit_transform( df[ 'top_similar' ] )
        lda = LatentDirichletAllocation( n_components = c.CLUSTERS, random_state = 19920917 )

        lda.fit( dtm )

        for index, topic in enumerate( lda.components_ ):
            print( index + 1 )
            print( f'The top 30 words for top #{ index + 1 }' )
            print( [ cv.get_feature_names()[ index ] for index in topic.argsort()[ -30: ] ] )
            print( '\n' )

        topic_results = lda.transform( dtm )
        df[ 'Descriptor' ] = topic_results.argmax( axis = 1 )

        print( "Define the " + str( c.CLUSTERS ) + " topics: \n" )
        topic  = ''
        topics = {
            'int_value' : [],
            'str_value' : []
        }

        for k in range( c.CLUSTERS ):
            topic = input( "What is topic #" + str( k + 1 ) + "? \n" )
            topics[ 'int_value' ].append( k )
            topics[ 'str_value' ].append( topic )

        new_labels = dict( zip( topics[ 'int_value' ], topics[ 'str_value' ] ) )

        df[ 'Descriptor' ].replace( new_labels, inplace = True )

        self.utilities.write_to_file( c.LABELED_DESCRIPTIONS, pd.DataFrame( df ) )