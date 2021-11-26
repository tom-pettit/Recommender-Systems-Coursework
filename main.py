import time
import csv
import json
from random import randint
import pandas as pd
from content_based_filter import ContentBasedSystem

# returns true if id is unique, and false otherwise
def checkNewID(id):
    with open('users.json', "r") as users_file:
        info = json.load(users_file)
        all_users = info['users']

        if id in all_users:
            return False
        else:
            return True

# returns names of the films given in a list of film IDs
def getMovieNames(ids):
    names = []
    movie_data = pd.read_csv('./data/movies.csv')

    for id in ids:
        title = movie_data[movie_data['movieId'] == id]['title'].iloc[0]
        names.append(title)

    return names

# shows active user their past ratings
def viewRatings(id):
    print('Loading your ratings...')
    info = pd.read_csv('./data/ratings.csv', index_col=False)

    ratings = (info.loc[info['userId'] == int(id)])

    movie_ids = ratings['movieId'].to_list()

    # print(movie_ids)

    movie_names = getMovieNames(movie_ids)

    result = ratings 

    result = result.drop('userId', 1)
    result = result.drop('timestamp', 1)
    result = result.drop('movieId', 1)

    result['movie'] = movie_names

    
    result.set_index('movie', inplace=True)


    print(result)
    
def showRecommendations(id):
    recommender = ContentBasedSystem(id)

    print('Creating your personalised recommendations...')
    predictions, user_top_tags = recommender.returnPredictedMovies()

    print('Because you like: ', [tag for tag in user_top_tags], '\n')

    print(predictions[['title', 'prediction', 'top_tags']])


# load system for the active user
def startSystem(id):
    print('Booting up the recommender systems for user: ', id, '... \n')
    # time.sleep(2)

    print('Menu: ')
    print('1. View Ratings')
    print('2. Add a New Rating')
    print('3. Recommendations')
    print('4. Logout')
    print('\n')

    choice = input()

    if (choice) == '1':
        viewRatings(id)
    elif choice == '3':
        showRecommendations(id)

def startUI():
    print('--------Song Recommender System--------')
    # time.sleep(3)
    print('Menu: ')
    print('1: Login')
    print('2: Create New User')
    print('Please choose an option: \n')

    choice = input()

    # login as an existing user
    if (choice) == '1':
        print('Login \n')
        # time.sleep(1)

        print('Please enter your User ID: ')

        userID = input()

        print('Checking if user exists... \n')
        # time.sleep(2)
        with open('users.json', "r") as users_file:
            info = json.load(users_file)
            existent_users = info['users']

            if int(userID) in existent_users:
                print('User exists. ')

                startSystem(userID)

            else:
                print('User does not exist in the userbase.')

        # time.sleep(1)

    # creating a new user
    elif choice == '2':
        print('Creating a New User... \n')
        # time.sleep(2)

        newID = randint(100000, 999999)

        users = []
        with open('users.json', "r") as users_file:
            info = json.load(users_file)
            users = info
            all_users = info['users']
            users = all_users.copy()

            while checkNewID(newID) is not True:
                newID = randint(100000, 999999)

            users.append(newID)

        to_write = {}
        to_write['users'] = users

        with open('users.json', "w") as users_file:
            to_write = json.dumps(to_write)
            users_file.write(to_write)

        print('Your new User ID is: ', newID)

        # time.sleep(1)

        print('Please remember this so you can login next time!')

def populateUsers():
    info = pd.read_csv('./data/ratings.csv', index_col=False)

    users = (info['userId'].unique())

    users = users.tolist()

    to_write = {}
    to_write['users'] = users

    # print(to_write)

    with open('users.json', "w") as users_file:
        to_write = json.dumps(to_write)
        users_file.write(to_write)


    print(users)


if __name__ == '__main__':
    startUI()
    # populateUsers()
