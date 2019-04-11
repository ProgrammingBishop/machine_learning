from ExploreMarketSegmentation import ExploreMarketSegmentation

import configurations as c

if __name__ == "__main__":
    explore_market = ExploreMarketSegmentation()

    # Generate Sparse Matrix
    explore_market.convert_to_sparse( c.FOLLOWER_DATA_CSV )