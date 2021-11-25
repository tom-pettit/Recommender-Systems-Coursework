import pandas as pd 

def createUserProfile(id):
    movieVectors = pd.read_csv('./data/genome-scores.csv')
    ratings = pd.read_csv('./data/ratings.csv', index_col=False)
    
    
    
    user_ratings = (ratings.loc[ratings['userId'] == int(id)])
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

    return user_profile

createUserProfile(1)