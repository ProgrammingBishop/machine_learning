# Blizzard Entertainment Market Segmentation using Tweepy Library
#### On hiatus for completion due to current obligations

## Progress
As of April 18, 2019 the parts of this project that have been completed are the following:

- Obtaining the target data
- Beginning to refactor the code into classes for reuse on other Twitter/Tweepy-related tasks. There is still much progress to be made on this part.
- Creating a barplot for users Blizzard Entertainment's followers also follow
- Creating the beginning feature space to be used for segmentation. This is currently a sparse matrix consisting of binary feature values reflecting on whether or not a Blizzard follower is following another user.
- The beginning class for K-Means that can display the elbow plot for the ideal K based on the above matrix specified, create k clusters, and create a label column for which cluster a user belongs to.

## TODO
When time is sufficient, the following are tasks to still be completed:

- Finalize refactoring of classes, the user interface through the command line, and the reuse of these classes for other Twitter/Tweepy-related tasks. Current code is operational, but not optimized enough for recycling, scalability, and maintainability.
- Finalize the feature space based on word association. This will be handled with the Natural Language Toolkit (NLTK) library. This will bring diversity to the current sparse matrix and most likely update the prototype labels set for the users. The additional features will be created based on user descriptions and location.
- Using the NLTK library to analyze tweets based on popular topics and sentiment of users (positive or negative), especially to understand which tweeted topics are most successful or least successful.
- Create a final report for these findings

## Purpose
Using Tweepy and Python the goal of this project is to understand Blizzard Entertainment's followers and engagement through the Twitter platform. 

#### The target information to retrieve include:

- Follower data, specifically screen name (for lookup in other methods), description, and location
- Blizzard Entertainment's profile tweets
- Tweets streaming in related to Blizzard Entertainment and various tracked topics related to Blizzard Entertainment's products
- The users that Blizzard Entertainment's followers are following

#### By leveraging this information, the following questions can be explored further:

- Who do followers of Blizzard Entertainment tend to also follow?
- What products of Blizzard Entertainment are followed? Which are popular or not so popular?
- Can the other users that Blizzard Entertainment's followers also follow be categorized by industry? Are they related to the gaming industry?
- Are there competitors based on who users also follow? If so are they similar or different to Blizzard Entertainment?
- What products, organizations, or celebrities can be targeted to endorse Blizzard Entertainment?
- Is there a reason some Blizzard products are followed more than others?
- How does the market diversify between segments?
- Are there similar interests or disinterests between segments?
- Are there opportunities from other users Blizzard Entertainment's followers also follow? Essentially what do they do different that Blizzard Entertainment is not doing but could be doing for increased user follower engagement?