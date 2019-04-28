from ExploreMarketSegmentation import ExploreMarketSegmentation
from PlotData                  import PlotData
from TextAnalysis              import TextAnalysis

class RunApplication():
    def __init__( self, steps, c ):
        explore_market = ExploreMarketSegmentation()
        plot_data      = PlotData()

        # Generate Barplot of Top N Followed Users
        # --------------------------------------------------
        if steps[0] == 't':
            print( "Creating barplot..." )
            plot_data.get_barplot_of_top_followed( c.TOP_FRIENDS_FOLLOWED_CSV, c.TOP_N )

        # Cluster Followers
        # --------------------------------------------------
        if steps[1] == 't':
            # Convert Descriptions to Vector of Top N Descriptors
            # Pass Descriptors and Followed Friends to Sparse Matrix Generator
            # Display Elbow and prompt to choose K
            # Cluster Users
            # Add Labels to Clustered
            return

        # Get Topic Modeling
        # --------------------------------------------------
        if steps[2] == 't':

        # Get Sentiment Analysis
        # --------------------------------------------------
        if steps[3] == 't':

        #     print( "Generating Sparse Matrix..." )
        #     explore_market.convert_to_sparse( c.FOLLOWER_FRIENDS_CSV, c.TOP_N )
        #     explore_market.find_optimal_k( c.SPARSE_FRIENDS_MATRIX_CSV, c.TOP_N )
        #     explore_market.create_cluster_labels( c.SPARSE_FRIENDS_MATRIX_CSV, c.CLUSTERS )