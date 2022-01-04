from surprise import SVD
import pandas as pd
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random

class CollaborativeFilteringSystem():
    # initialise id, the SVD model and the training and testing data variables used when evaluating the model
    def __init__(self, id):
        self.id = int(id)
        self.svd = SVD() 

    # prepare the dataset in a format suitable for evaluating the model
    def prepareDataset(self):
        ratings = pd.read_csv('./data/ratings.csv')
        ratings = ratings.drop(columns='timestamp')

        reader = Reader()
        data = Dataset.load_from_df(ratings, reader)

        print('prepared dataset')
        return data

    # split the dataset into train and test datasets
    def trainTestSplit(self, data):
        # split into train and test sets with 7.5% of total data in train and 2.5% in test. This is because the model is very large and takes a long time to run on large amounts of data
        training_data, testing_data = train_test_split(data, train_size=0.075, test_size=.025)

        print('split into train and test sets')
        return training_data, testing_data

    # train the SVD model. This takes a while. A model has already been run and saved to trained_svd.sav.
    def trainModel(self):
        # prepare the dataset
        data = self.prepareDataset()

        # split into train and test datasets
        training_data, testing_data = self.trainTestSplit(data)

        print('fitting svd')
        # fit the SVD model on the training data
        self.svd.fit(training_data)

        print('svd trained successfully')

        # save the model to a file so does not need to be trained each time
        model_file = 'trained_svd.sav'
        pickle.dump(self.svd, open(model_file, 'wb'))
        print('saved svd model to sav file')

    # load in the pre-trained model
    def loadModelFromFile(self, filename):
        self.svd = pickle.load(open(filename, 'rb'))

    # calculate the RMSE of the SVD model
    def evaluateModel(self, testing_data):
        # make predictions on the testing dataset. This returns a Prediction object, which includes the predicted ratings and the actual ratings
        predictions = self.svd.test(testing_data)

        print('made predictions')

        # use the accuracy function from scikit-surprise to calculate RMSE
        rmse = accuracy.rmse(predictions)

        return rmse

    # make recommendations
    def makePredictions(self):
        movies = pd.read_csv('./data/movies.csv')
        # create a new column to store prediction rating for each movie
        movies['prediction'] = [0 for i in range(movies.shape[0])]

        ratings = pd.read_csv('./data/ratings.csv', index_col=False)
        all_tags = pd.read_csv('./data/genome-scores.csv')
        all_tags_info = pd.read_csv('./data/genome-tags.csv')

        # make predictions on each movie using the SVD model, and store it in the prediction (index 3) column
        for index, row in movies.iterrows():
            prediction = self.svd.predict(self.id, row['movieId']).est
            movies.iloc[index, 3] = prediction 

        # find all ratings by active user
        user_ratings = (ratings.loc[ratings['userId'] == int(self.id)])
        user_ratings = user_ratings.drop('userId', axis=1)
        user_ratings.set_index('movieId', inplace=True)

        # array of movie IDs the user has seen
        seen_movies = user_ratings.index.to_list()

        # get rid of movies the user has seen already
        movies = movies.loc[~movies['movieId'].isin(seen_movies)]

        # apply Min-Max Feature Scaling to the predictions
        min_max_scaler = MinMaxScaler()
        movies[['prediction']] = min_max_scaler.fit_transform(movies[['prediction']])

        # sort the dataframe by the prediction values in descending order
        movies = movies.sort_values('prediction', ascending=False)

        movies.set_index('movieId', inplace=True)

        # get top 10 predictions
        top_10_predictions = movies[:10]

        # get the top 5 tags and the title for each of the top 10 predictions
        top_tags = []
        for index, row in top_10_predictions.iterrows():
            id = index
            movie = movies.loc[movies.index == id]
            # get the title of each movie
            title = movie['title'].item()

            movie_tags = all_tags.loc[all_tags['movieId'] == id]

            # sort the tags by the relevance in descending order
            movie_tags = movie_tags.sort_values('relevance', ascending=False)

            # get the top 5 tags by relevance score
            top_5_tag_ids = (movie_tags[:5])['tagId'].to_list()

            # get the names for each of the top 5 tags
            tag_names = []
            for tag in top_5_tag_ids:
                tag_name = all_tags_info.loc[all_tags_info['tagId'] == tag]['tag'].item()
                tag_names.append(tag_name)

            # if there are no tags to display
            if len(tag_names) == 0:
                tag_names = "No tags to display"
            top_tags.append(tag_names)

        # add the top tags into the dataframe
        top_10_predictions['top_tags'] = top_tags


        return top_10_predictions

    # shows the details of the model. Used in testing.
    def viewModelDetails(self):
        user_factors = self.svd.pu
        item_factors = self.svd.qi
        user_biases = self.svd.bu
        item_biases = self.svd.bi

        print(user_biases)
        print(item_biases)

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
    print('Running evaluations...')
    ratings = pd.read_csv('./data/ratings.csv')
    random_user = random.choice(ratings['userId'].unique())
    model = CollaborativeFilteringSystem(random_user)
    model.loadModelFromFile('trained_svd.sav')

    data = model.prepareDataset()
    training_data, testing_data = model.trainTestSplit(data)
    rmse = model.evaluateModel(testing_data)

    predictions = model.makePredictions()
    diversity = model.calculateDiversity(predictions)

    print('Diversity: ', diversity)