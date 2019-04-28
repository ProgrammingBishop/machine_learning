import pandas         as pd
import configurations as c

from nltk.tokenize import word_tokenize
from Utilities     import Utilities


class TextAnalysis():
    utilities = Utilities()

    def tokenize_description( self, filepath ):
        '''
        Returns :tokenize follower descriptions
        --------------------------------------------------
        filepath : location to obtain data
        '''
        tokens = {
            'screen_name'   : [],
            'tokenize_desc' : []
        }

        follower_data = pd.read_csv( filepath )

        for _, follower in follower_data.iterrows():
            tokens[ 'screen_name'   ].append( follower[ 'screen_name' ] )
            tokens[ 'tokenize_desc' ].append( word_tokenize( follower[ 'description' ] ) )

        self.utilities.write_to_file( c.TOKENIZED_DESCRIPTIONS, pd.DataFrame( tokens ) )

    def map_reduce( self ):
        return

    def tokens_to_sparse_matrix( self ):
        return