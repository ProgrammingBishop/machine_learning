from ExploreMarketSegmentation import ExploreMarketSegmentation
from PlotData                  import PlotData
from TextAnalysis              import TextAnalysis
from GetDescriptors            import GetDescriptors
import sys

class RunApplication():
    def __init__( self, steps, c ):
        explore_market = ExploreMarketSegmentation()
        plot_data      = PlotData()
        get_desc       = GetDescriptors()

        # Generate Barplot of Top N Followed Users
        # --------------------------------------------------
        if steps[0] == 't':
            print( "Creating barplot...\n" )
            plot_data.get_barplot_of_top_followed( c.TOP_FRIENDS_FOLLOWED_CSV, c.TOP_N )


        # Cluster Followers
        # --------------------------------------------------
        if steps[1] == 't':
            print( "Beginning K-Means Step...\n" )
            get_k = ''

            while True:
                try:
                    get_k = str( input( "Do you have an optimal K" + " (t / f): \n" ) ).lower()

                    if get_k == 't' or get_k == 'f':
                        if get_k == 'f':
                            explore_market.find_optimal_k( c.SPARSE_FRIENDS_MATRIX_CSV, c.TOP_N )
                            continue
                        else:
                            user_input = input( "Enter the optimal value for K: \n" )

                            while True:
                                try:
                                    c.CLUSTERS = int( user_input )
                                    get_desc.get_descriptors()
                                    get_desc.topic_model_descriptors()
                                    break
                                except:
                                    print( "Error" )
                                    sys.exit()
                            break    
                except:
                    print( "Something went wrong..." )
                    sys.exit()


        # Get Topic Modeling
        # --------------------------------------------------
        if steps[2] == 't':
            return


        # Get Sentiment Analysis
        # --------------------------------------------------
        if steps[3] == 't':
            return

        #     print( "Generating Sparse Matrix..." )
        #     explore_market.convert_to_sparse( c.FOLLOWER_FRIENDS_CSV, c.TOP_N )
        #     explore_market.find_optimal_k( c.SPARSE_FRIENDS_MATRIX_CSV, c.TOP_N )
        #     explore_market.create_cluster_labels( c.SPARSE_FRIENDS_MATRIX_CSV, c.CLUSTERS )