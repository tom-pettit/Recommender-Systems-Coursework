import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from surprise import Reader, Dataset
from sklearn.model_selection import train_test_split
from surprise import accuracy
import random
from sklearn.metrics import mean_squared_error
import numpy as np

class ContentBasedSystem():
    def __init__(self, id):
        self.id = id

    # create user profile for user
    def createUserProfile(self):
        movieVectors = pd.read_csv('./data/genome-scores.csv')
        ratings = pd.read_csv('./data/ratings.csv', index_col=False)
        
        # get all ratings by user
        user_ratings = (ratings.loc[ratings['userId'] == int(self.id)])
        user_ratings = user_ratings.drop('userId', axis=1)
        user_ratings.set_index('movieId', inplace=True)

        # get all movies the user has rated
        movie_ids = user_ratings.index.to_list()

        # get movie keyword vector for the rated movies
        user_movies = movieVectors.loc[movieVectors['movieId'].isin(movie_ids)]
        user_movies = user_movies.pivot(index='movieId', columns='tagId', values='relevance')

        # if the user has no ratings, assume they have no preference in tags ie they rate every tag as equally important
        if len(movie_ids) == 0:
            # give all 1128 tags a weighting of 1, so they are all equally important
            user_profile = [1 for i in range(1128)]

            # convert into pandas dataframe
            user_profile = pd.DataFrame([user_profile], columns=[i for i in range(1, 1129)])
            return user_profile, movie_ids

        # add user ratings to the movie keyword vectors
        user_movies['rating'] = user_ratings['rating']

        # calculate average rating for active user
        user_avg_rating = user_movies['rating'].mean()

        # if the user has rated 1 or fewer films, don't subtract the mean rating
        number_of_ratings = user_movies.shape[0]

        if number_of_ratings >= 5:
            # adjust ratings for mean rating of active user
            user_movies['rating'] = user_movies['rating'] - user_avg_rating
        else:
            pass

        # create user profile from dot product of each keyword column with the ratings for each movie
        user_profile = []
        for i in range(len(user_movies.columns)-1):
            value = user_movies.iloc[:, i].dot(user_movies['rating'])
            user_profile.append(value)

        # convert into Pandas Dataframe
        user_profile = pd.DataFrame([user_profile], columns=[i for i in range(1, len(user_movies.columns))])

        return user_profile, movie_ids

    # create a user profile, but for evaluation purposes
    def createTestingUserProfile(self, id, ratings):
        # feed in a specified ID, and a set of ratings, in order for the system to be evaluated properly
        # otherwise, this code is exactly the same as the createUserProfile method above
        movieVectors = pd.read_csv('./data/genome-scores.csv')
        
        user_ratings = (ratings.loc[ratings['userId'] == int(id)])
        user_ratings = user_ratings.drop('userId', axis=1)
        user_ratings.set_index('movieId', inplace=True)
        movie_ids = user_ratings.index.to_list()

        user_movies = movieVectors.loc[movieVectors['movieId'].isin(movie_ids)]
        user_movies = user_movies.pivot(index='movieId', columns='tagId', values='relevance')

        user_movies['rating'] = user_ratings['rating']

        user_avg_rating = user_movies['rating'].mean()

        number_of_ratings = user_movies.shape[0]

        if number_of_ratings >= 2:
            user_movies['rating'] = user_movies['rating'] - user_avg_rating
        else:
            pass

        user_profile = []
        for i in range(len(user_movies.columns)-1):
            value = user_movies.iloc[:, i].dot(user_movies['rating'])
            user_profile.append(value)

        user_profile = pd.DataFrame([user_profile], columns=[i for i in range(1, len(user_movies.columns))])

        return user_profile, movie_ids

    # make recommendations
    def makePredictions(self):
        # create user profile
        user_profile, seen_movies = self.createUserProfile()

        movieVectors = pd.read_csv('./data/genome-scores.csv')
        movieVectors = movieVectors.pivot(index='movieId', columns='tagId', values='relevance')

        # creat weighted keyword vector by multiplying all movie keyword vectors by the user profile
        weighted_movies = pd.DataFrame()
        for i in range(len(movieVectors.columns)):
            weighted_movies[i] = movieVectors.iloc[:,i].mul(user_profile.iloc[0,i])

        # prediction for each movie is calculated from sum of this weighted keyword vector
        weighted_movies['prediction'] = weighted_movies[list(weighted_movies.columns)].sum(axis=1)

        # apply Min-Max Feature Scaling
        min_max_scaler = MinMaxScaler()

        weighted_movies[['prediction']] = min_max_scaler.fit_transform(weighted_movies[['prediction']])

        # sort the dataframe by prediction score in descending order
        weighted_movies = weighted_movies.sort_values('prediction', ascending=False)

        # get rid of any films that the user has seen previously
        weighted_movies = weighted_movies.loc[~weighted_movies.index.isin(seen_movies)]
        
        # get the top 10 predictions
        top_10_predictions = weighted_movies[:10]

        # get tags for each of top 10 predictions
        tags = user_profile.columns
        features = user_profile.values.flatten().tolist()

        # what tags are most important to the user, as shown in their user profile
        user_tag_impact_dict =  dict(zip(tags, features))

        # sort by tag score in descending order
        user_tag_impact_dict = dict(sorted(user_tag_impact_dict.items(), key=lambda item: item[1], reverse=True))

        # get top 10 tags for the user
        top_user_tags = {k: user_tag_impact_dict[k] for k in list(user_tag_impact_dict.keys())[:10]}

        # top 10 tag IDs for user
        top_user_tags = list(top_user_tags.keys())

        tags_info = pd.read_csv('./data/genome-tags.csv')

        # get the name for the top 10 tag IDs
        user_top_tags_info = tags_info.loc[tags_info['tagId'].isin(top_user_tags)]
        user_top_tags = list(user_top_tags_info['tag'])

        return top_10_predictions, user_top_tags, seen_movies

    # make recommendations for evaluation purposes
    def makeTestingPredictions(self, id, train_ratings, test_ratings):
        # create user profile, but only off of supplied training ratings
        user_profile, seen_movies = self.createTestingUserProfile(id, train_ratings)
        movieVectors = pd.read_csv('./data/genome-scores.csv')
        movieVectors = movieVectors.pivot(index='movieId', columns='tagId', values='relevance')

        # create movieVectors only from supplied test ratings, not entire dataset
        test_movie_ids = test_ratings['movieId'].unique()

        # some movies have been reviewed but do not have genome scores
        # thus, for testing purposes, these will be removed from the testing dataset, as the model cannot accurately predict a prediction rating for these movies
        
        # movies with genome scores and so accurate predictions
        ids_wanted = []

        # movies with no genome scores and so will be removed from test set
        ids_unwanted = []
        for movieId in test_movie_ids:
            if movieId in movieVectors.index:
                ids_wanted.append(movieId)
            else:
                ids_unwanted.append(movieId)

        movieVectors = movieVectors.loc[movieVectors.index.isin(ids_wanted)]

        # create weighted movie keyword vector by multiplying movie keyword vectors with user profile
        weighted_movies = pd.DataFrame()

        for i in range(len(movieVectors.columns)):
            weighted_movies[i] = movieVectors.iloc[:,i].mul(user_profile.iloc[0,i])

        # create prediction from summing weighted keyword vector for each movie
        weighted_movies['prediction'] = weighted_movies[list(weighted_movies.columns)].sum(axis=1)


        # apply Min-Max Feature Scaling
        min_max_scaler = MinMaxScaler()


        weighted_movies[['prediction']] = min_max_scaler.fit_transform(weighted_movies[['prediction']])

        # upscale predictions to be a score between 0 and 5, for evaluation purposes
        weighted_movies['prediction'] = weighted_movies['prediction'] * 5

        # round prediction scores to nearest 0.5, again for evaluation purposes
        weighted_movies['prediction'] = round(weighted_movies['prediction'] * 2) / 2

        predictions =  weighted_movies[['prediction']]


        return predictions, ids_unwanted

    # return predictions in a more appealing way to active user
    def returnPredictedMovies(self):
        # create predictions and top tags for the active user
        top_10_predictions, user_top_tags, seen_movies = self.makePredictions()
        movieDetails = pd.read_csv('./data/movies.csv')
        all_tags = pd.read_csv('./data/genome-scores.csv')
        all_tags_info = pd.read_csv('./data/genome-tags.csv')

        # get titles and top tags for movies in top 10
        titles = []
        top_tags = []
        for index, row in top_10_predictions.iterrows():
            id = index
            movie = movieDetails.loc[movieDetails['movieId'] == id]
            # get the title of the movie
            title = movie['title'].item()
            titles.append(title)

            # get all tags of the movie
            movie_tags = all_tags.loc[all_tags['movieId'] == id]

            # sort the tags by their relevance scores
            movie_tags = movie_tags.sort_values('relevance', ascending=False)

            # list of top 5 tag IDs
            top_5_tag_ids = (movie_tags[:5])['tagId'].to_list()

            # get the names of the top 5 tag IDs
            tag_names = []
            for tag in top_5_tag_ids:
                tag_name = all_tags_info.loc[all_tags_info['tagId'] == tag]['tag'].item()
                tag_names.append(tag_name)

            top_tags.append(tag_names)

        # add title and top 5 tags for each movie to the dataframe
        top_10_predictions['title'] = titles
        top_10_predictions['top_tags'] = top_tags

        return top_10_predictions, user_top_tags, seen_movies

    # split data into training and testing datasets
    def trainTestSplit(self, data):
        # split into train and test datasets with 25% in test and 75% in train
        training_data, testing_data = train_test_split(data, test_size=0.25)

        return training_data, testing_data

    # evaluate the model accuracy using RMSE 
    def evaluateModel(self):
        ratings = pd.read_csv('./data/ratings.csv')
        ratings = ratings.drop(columns='timestamp')

        # calculate the RMSE for the model by selecting 5 users at random and calculating the RMSE for their actual and predicted movie ratings
        rmse_sum = 0
        for i in range(5):
            # select a random user
            random_user = random.choice(ratings['userId'].unique())

            # get all their ratings
            user_ratings = ratings.loc[ratings['userId'] == random_user]

            # split data into train and testing datasets
            training_data, testing_data = self.trainTestSplit(user_ratings)

            # make predictions on testing data, by creating user profile from training data and calculating predictions on the testing data
            predictions, unwanted_ids = self.makeTestingPredictions(random_user, training_data, testing_data)

            # change format of testing data dataframe
            testing_data.set_index('movieId', inplace=True)

            # remove movies that have no genome scores from test set
            testing_data = testing_data.loc[~testing_data.index.isin(unwanted_ids)]

            testing_data = testing_data.drop('userId', axis=1)

            # sort by movie ID
            testing_data = testing_data.sort_values('movieId', ascending=True)

            # print the actual ratings on the testing dataset movies
            # print(testing_data)

            # print the predicted ratings on the testing dataset movies
            # print(predictions)

            # calculate the RMSE 
            # print(testing_data['rating'])
            # print(predictions['prediction'])
            rmse = mean_squared_error(testing_data['rating'], predictions['prediction'], squared=False)

            # print(rmse)

            # append RMSE to sum 

            rmse_sum += rmse 

        # calculate average RMSE across 5 random users
        rmse = rmse_sum / 5

        return rmse

    # calculate cosine similarity between 2 movies with id1 and id2
    def CosineSimilarity(self, ratings, id1, id2):
        # determine which users rated movie with id1 and which users rated movie with id2
        users_rated_1 = ratings.loc[ratings['movieId'] == id1]['userId']
        users_rated_2 = ratings.loc[ratings['movieId'] == id2]['userId']

        # get number of users who rated movie with id1 and movie with id2
        count_users_rated_1 = users_rated_1.shape[0]
        count_users_rated_2 = users_rated_2.shape[0]

        # find the number of users who rated both movie with id1 and movie with id2
        count_users_rated_both = len(list(set(users_rated_1).intersection(users_rated_2)))

        # calculate cosine similarity between the two movies
        cosine_similarity = count_users_rated_both / ( np.sqrt(count_users_rated_1) * np.sqrt(count_users_rated_2) )

        return cosine_similarity

    # calculate the diversity score for the system
    def calculateDiversity(self, predictions):
        predictions = predictions[['prediction']]
        ratings = pd.read_csv('./data/ratings.csv')
        
        # calculate cosine similarities between every pair of predicted movies 
        cosine_similarities = []
        for index1, row1 in predictions.iterrows():
            for index2, row2 in predictions.iterrows():
                # index is the movie ID
                # so ensure movies are different
                if index1 != index2:
                    cosine_similarity = self.CosineSimilarity(ratings, index1, index2)
                    cosine_similarities.append(cosine_similarity)

        # create average cosine similarity between all pairs of movies
        avg_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)

        # diversity is calculated using this formula
        diversity = 1 - avg_cosine_similarity

        return diversity



if __name__ == '__main__':
    # run the file to run evaluations
    print('Running evaluations...')
    ratings = pd.read_csv('./data/ratings.csv')
    # select a random user
    random_user = random.choice(ratings['userId'].unique())

    # set up the model using this random user
    model = ContentBasedSystem(random_user)

    # calculate RMSE
    rmse = model.evaluateModel()
    print('RMSE: ', rmse)

    # calculate Diversity
    predictions, _, __ = model.makePredictions()
    diversity = model.calculateDiversity(predictions)
    print('Diversity: ', diversity)