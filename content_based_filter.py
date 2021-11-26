import pandas as pd 
from sklearn.preprocessing import MinMaxScaler


class ContentBasedSystem():
    def __init__(self, id):
        self.id = id

    def createUserProfile(self):
        movieVectors = pd.read_csv('./data/genome-scores.csv')
        ratings = pd.read_csv('./data/ratings.csv', index_col=False)
        
        
        
        user_ratings = (ratings.loc[ratings['userId'] == int(self.id)])
        user_ratings = user_ratings.drop('userId', axis=1)
        user_ratings.set_index('movieId', inplace=True)
        movie_ids = user_ratings.index.to_list()

        user_movies = movieVectors.loc[movieVectors['movieId'].isin(movie_ids)]
        user_movies = user_movies.pivot(index='movieId', columns='tagId', values='relevance')

        user_movies['rating'] = user_ratings['rating']

        user_avg_rating = user_movies['rating'].mean()

        user_movies['rating'] = user_movies['rating'] - user_avg_rating

        user_profile = []
        for i in range(len(user_movies.columns)-1):
            value = user_movies.iloc[:, i].dot(user_movies['rating'])
            user_profile.append(value)

        user_profile = pd.DataFrame([user_profile], columns=[i for i in range(1, len(user_movies.columns))])

        return user_profile, movie_ids



    def makePredictions(self):
        user_profile, seen_movies = self.createUserProfile()
        movieVectors = pd.read_csv('./data/genome-scores.csv')
        movieVectors = movieVectors.pivot(index='movieId', columns='tagId', values='relevance')

        weighted_movies = pd.DataFrame()

        for i in range(len(movieVectors.columns)):
            weighted_movies[i] = movieVectors.iloc[:,i].mul(user_profile.iloc[0,i])

        weighted_movies['prediction'] = weighted_movies[list(weighted_movies.columns)].sum(axis=1)

        min_max_scaler = MinMaxScaler()


        weighted_movies[['prediction']] = min_max_scaler.fit_transform(weighted_movies[['prediction']])

        weighted_movies = weighted_movies.sort_values('prediction', ascending=False)

        weighted_movies = weighted_movies.loc[~weighted_movies.index.isin(seen_movies)]
        
        top_10_predictions = weighted_movies[:10]

        tags = user_profile.columns
        features = user_profile.values.flatten().tolist()

        user_tag_impact_dict =  dict(zip(tags, features))

        user_tag_impact_dict = dict(sorted(user_tag_impact_dict.items(), key=lambda item: item[1], reverse=True))

        top_user_tags = {k: user_tag_impact_dict[k] for k in list(user_tag_impact_dict.keys())[:5]}

        top_user_tags = list(top_user_tags.keys())

        tags_info = pd.read_csv('./data/genome-tags.csv')

        user_top_tags_info = tags_info.loc[tags_info['tagId'].isin(top_user_tags)]

        user_top_tags = list(user_top_tags_info['tag'])

        return top_10_predictions, user_top_tags

    def returnPredictedMovies(self):
        top_10_predictions, user_top_tags = self.makePredictions()
        movieDetails = pd.read_csv('./data/movies.csv')
        all_tags = pd.read_csv('./data/genome-scores.csv')
        all_tags_info = pd.read_csv('./data/genome-tags.csv')

        titles = []
        top_tags = []
        for index, row in top_10_predictions.iterrows():
            id = index
            movie = movieDetails.loc[movieDetails['movieId'] == id]
            title = movie['title'].item()
            titles.append(title)

            movie_tags = all_tags.loc[all_tags['movieId'] == id]

            movie_tags = movie_tags.sort_values('relevance', ascending=False)

            top_5_tag_ids = (movie_tags[:5])['tagId'].to_list()

            tag_names = []
            for tag in top_5_tag_ids:
                tag_name = all_tags_info.loc[all_tags_info['tagId'] == tag]['tag'].item()
                tag_names.append(tag_name)

            top_tags.append(tag_names)

        top_10_predictions['title'] = titles
        top_10_predictions['top_tags'] = top_tags

        # print(top_10_predictions[['title', 'prediction']])

        return top_10_predictions, user_top_tags

        