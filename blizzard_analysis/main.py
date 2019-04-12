from ExploreMarketSegmentation import ExploreMarketSegmentation

import configurations as c

if __name__ == "__main__":
    explore_market = ExploreMarketSegmentation()

    # Generate Sparse Matrix
    # explore_market.convert_to_sparse( c.FOLLOWER_FRIENDS_CSV, c.TOP_N )
    # explore_market.get_barplot_of_top_followed( c.TOP_FRIENDS_FOLLOWED_CSV, c.TOP_N )