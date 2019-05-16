# Imports
# ==================================================
from scipy                           import spatial
from Utilities                        import Utilities 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition           import LatentDirichletAllocation

import spacy
import pandas         as pd
import configurations as c


# Class
# ==================================================
class GetDescriptors():
    # PRIVATE
    __util = '' 
    __punc = ''
    __nlp  = ''
    __str  = ''
    __stop = ''
    
    def __init__( self ):
        self.__util = Utilities()
        self.__nlp  = spacy.load( 'en_core_web_lg' )
        self.__stop = spacy.lang.en.stop_words.STOP_WORDS
        self.__punc = '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n '


    def __clean_text( self, description ):
        sentence = []

        for token in self.__nlp( description ):
            if token.text not in self.__punc:
                if token.text not in self.__stop:
                    if token.text in self.__nlp.vocab:
                        sentence.append( token.text )

        return " ".join( sentence )


    # PUBLIC
    def segment_descriptions( self ):
        data = pd.read_csv( c.STREAM_DATAFRAME_CSV )
        
        data.fillna( '', inplace = True )
        data[ 'description' ] = [ self.__clean_text( desc ) for desc in data[ 'description' ] ]

        self.__util.write_to_file( c.STREAM_DATAFRAME_CSV, pd.DataFrame( data ), 'w' )


    # TODO Refactor This
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

        self.__util .write_to_file( c.LABELED_DESCRIPTIONS, pd.DataFrame( df ) )