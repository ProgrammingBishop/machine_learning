from PlotData          import PlotData
from TweepyKMeans      import TweepyKMeans
from GetDescriptors    import GetDescriptors
from DefineApplication import DefineApplication
from Utilities         import Utilities
from sys               import exit


class RunApplication():
    # PRIVATE
    def __init__( self, steps, c ):
        util       = Utilities()
        plot_data  = PlotData()
        tweepy_k   = TweepyKMeans()
        get_desc   = GetDescriptors()
        define_app = DefineApplication()

        # Generate Barplot Step
        # --------------------------------------------------
        if steps[0] == 't':
            print( "Beginning the barplot generation step..." )
            print( '--------------------------------------------------\n\n' )

            try:
                plot_data.get_barplot_pdf( c )

            except:
                util.finding_file_error( 'TOP_FRIENDS_FOLLOWED_CSV', 'popular_friends_to_csv()' )


        # Cluster Followers Step
        # --------------------------------------------------
        if steps[1] == 't':
            print( "Beginning the clustering step..." )
            print( '--------------------------------------------------\n\n' )

            while True:
                try:
                    get_k = str( input( "Do you have an optimal K value to use? (t / f): " ) ).lower()
                    print( '--------------------------------------------------\n\n' )

                    if get_k == 'f':
                        print( "Generating a sparse matrix to cluster for optimal K..." )
                        print( '--------------------------------------------------\n\n' )
                        
                        try:
                            tweepy_k.find_optimal_k( c )
                            continue

                        except:
                            util.finding_file_error( 'FOLLOWER_FRIENDS_CSV', 'follower_friends_to_csv()' )

                            
                    if get_k == 't':
                        user_input = input( "Enter the optimal value for K: " )
                        print( '--------------------------------------------------\n\n' )

                        try:
                            c.CLUSTERS = int( user_input )

                            # TODO: Update GetDescriptors class
                            get_desc.get_descriptors()
                            get_desc.topic_model_descriptors()
                            break

                        except:
                            util.finding_file_error( 'STREAM_DATAFRAME_CSV', 'stream_to_csv()' )

                except:
                    print( "Something went wrong..." )
                    exit()


        # Get Topic Modeling
        # --------------------------------------------------
        # if steps[2] == 't':
        #     return


        # Get Sentiment Analysis
        # --------------------------------------------------
        # if steps[3] == 't':
        #     return

        #     print( "Generating Sparse Matrix..." )
        #     tweepy_k.convert_to_sparse( c.FOLLOWER_FRIENDS_CSV, c.TOP_N )
        #     tweepy_k.create_cluster_labels( c.SPARSE_FRIENDS_MATRIX_CSV, c.CLUSTERS )