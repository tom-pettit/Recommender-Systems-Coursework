from surprise import SVD
import pandas as pd
from surprise.model_selection import cross_validate
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle
from sklearn.preprocessing import MinMaxScaler

class CollaborativeFilteringSystem():
    def __init__(self, id):
        self.id = int(id)
        self.svd = SVD() 
        self.training_data = 0
        self.testing_data = 0

    def prepareDataset(self):
        ratings = pd.read_csv('./data/ratings.csv')
        ratings = ratings.drop(columns='timestamp')

        reader = Reader()
        data = Dataset.load_from_df(ratings, reader)

        print('prepared dataset')
        return data

    def trainTestSplit(self, data):
        training_data, testing_data = train_test_split(data, test_size=.25)

        self.training_data = training_data
        self.testing_data = testing_data

        print('split into train and test sets')
        return training_data, testing_data


    def trainModel(self):
        data = self.prepareDataset()
        training_data, testing_data = self.trainTestSplit(data)

        print('fitting svd')
        self.svd.fit(training_data)

        print('svd trained successfully')

        model_file = 'trained_svd.sav'
        pickle.dump(self.svd, open(model_file, 'wb'))
        print('saved svd model to sav file')

    def loadModelFromFile(self, filename):
        self.svd = pickle.load(open(filename, 'rb'))

    def makeTestingPredictions(self):
        print(self.svd)
        predictions = self.svd.test(self.testing_data)

        print('made predictions')
        accuracy.rmse(predictions)

    def makePredictions(self):
        movies = pd.read_csv('./data/movies.csv')
        movies['prediction'] = [0 for i in range(movies.shape[0])]
        ratings = pd.read_csv('./data/ratings.csv', index_col=False)

        for index, row in movies.iterrows():
            prediction = self.svd.predict(self.id, row['movieId']).est
            movies.iloc[index, 3] = prediction 


        user_ratings = (ratings.loc[ratings['userId'] == int(self.id)])
        user_ratings = user_ratings.drop('userId', axis=1)
        user_ratings.set_index('movieId', inplace=True)
        seen_movies = user_ratings.index.to_list()

        movies = movies.loc[~movies['movieId'].isin(seen_movies)]

        min_max_scaler = MinMaxScaler()


        movies[['prediction']] = min_max_scaler.fit_transform(movies[['prediction']])


        movies = movies.sort_values('prediction', ascending=False)

        movies.set_index('movieId', inplace=True)

        top_10_predictions = movies[:10]

        # print(top_10_predictions)

        return movies[:10]

    def viewModelDetails(self):
        user_factors = self.svd.pu
        item_factors = self.svd.qi
        user_biases = self.svd.bu
        item_biases = self.svd.bi

        print(user_biases)
        print(item_biases)


# collaborative_filter = CollaborativeFilteringSystem(1)

# # data = collaborative_filter.prepareDataset()

# # training_data, testing_data = collaborative_filter.trainTestSplit(data)

# # collaborative_filter.trainModel(training_data)

# collaborative_filter.loadModelFromFile('trained_svd.sav')
# collaborative_filter.makePredictions()

# collaborative_filter.viewModelDetails()