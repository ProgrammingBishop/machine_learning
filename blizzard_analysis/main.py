from ExploreMarketSegmentation import ExploreMarketSegmentation
from PlotData                  import PlotData
from TextAnalysis              import TextAnalysis

import configurations as c

if __name__ == "__main__":
    explore_market = ExploreMarketSegmentation()
    plot_data      = PlotData()
    text_analysis  = TextAnalysis()

    # Generate Sparse Matrix
    # explore_market.convert_to_sparse( c.FOLLOWER_FRIENDS_CSV, c.TOP_N )

    # Generate Bar Plot of Top-most Followed Friends
    # plot_data.get_barplot_of_top_followed( c.TOP_FRIENDS_FOLLOWED_CSV, c.TOP_N )

    # Add Descriptions Tokens to Sparse\
    text_analysis.tokenize_description( c.FOLLOWER_DATA_CSV )

    # Find Optimal K and Plot K-Means
    # explore_market.find_optimal_k( c.SPARSE_FRIENDS_MATRIX_CSV, c.TOP_N )
    # explore_market.create_cluster_labels( c.SPARSE_FRIENDS_MATRIX_CSV, c.CLUSTERS )