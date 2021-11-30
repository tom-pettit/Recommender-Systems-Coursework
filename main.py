import time
import csv
import json
from random import randint
import pandas as pd
from content_based_filter import ContentBasedSystem
from collaborative_filter import CollaborativeFilteringSystem
import datetime
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'


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

# let the active user leave another review
def addNewRating(id):
    movies = pd.read_csv('./data/movies.csv')
    ratings = pd.read_csv('./data/ratings.csv')

    print('Please note: When you leave a rating for a film, we save this to our database. This data is then used to make better recommendations for you in future.')
    
    movie_id = int(input('Please enter the ID of the movie you would like to leave a rating for: '))

    movie = movies.loc[movies['movieId'] == movie_id]
    existing_entry = np.where((ratings['userId'] == int(id)) & (ratings['movieId'] == movie_id))

    if len(existing_entry[0]) == 0:
        print('\n')
        print('You are choosing to leave a rating for the film: ', movie['title'].item())
        print('\n')
        rating = float(input('Please leave a rating for this film (out of 5, in increments of 0.5): '))

        timestamp = datetime.datetime.now().timestamp()

        to_write = {'userId':id, 'movieId':movie_id, 'rating':rating, 'timestamp':timestamp}

        with open('./data/ratings.csv', 'a', newline='') as csvfile:
            fieldnames = ['userId','movieId', 'rating', 'timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writerow(to_write)

        print('\n')
        print('Successfully left your review... \n')

    else:
        print('You have already left a review for the film: ', movie['title'].item())
        print('\n')

    mainMenu(id)

# allows active user to edit a previous rating of theirs
def editRating(id):
    ratings = pd.read_csv('./data/ratings.csv')
    movies = pd.read_csv('./data/movies.csv')


    movie_id = int(input('Enter the ID of the movie you would like to edit your rating for: \n'))

    rating = ratings.loc[(ratings['userId'] == int(id)) & (ratings['movieId'] == movie_id)]

    movie_info = movies.loc[movies['movieId'] == movie_id]

    if rating.empty:
        print('You have not left a review for the film: ', movie_info['title'].item(), '\n')
    else:

        print('Film: ', movie_info['title'].item(), '\n')

        print('Previous Rating: ', rating['rating'].item(), '\n')

        new_rating = float(input('Enter a new rating for this film: '))
        print('\n')

        index = np.where((ratings['userId'] == int(id)) & (ratings['movieId'] == movie_id))[0][0]

        ratings.iloc[index, 2] = new_rating

        print('Updating the rating in the dataset...')

        print('WARNING: This may take some time \n')

        ratings.to_csv('./data/ratings.csv', index=False)

        print('\n Successfully updated your rating \n')

    mainMenu(id)

# shows active user their past ratings
def viewRatings(id):
    print('Loading your ratings... \n')
    info = pd.read_csv('./data/ratings.csv', index_col=False)

    ratings = (info.loc[info['userId'] == int(id)])

    if ratings.empty:
        print('No ratings to display \n')
    else:

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

        print('\n')

    mainMenu(id)

# prints recommendations  
def showRecommendations(id):
    recommender_choice = int(input('1: Content-Based Filter or 2: Collaborative Filter \n'))

    if recommender_choice == 1:
        recommender = ContentBasedSystem(id)

        print('Creating your personalised recommendations...')
        predictions, user_top_tags = recommender.returnPredictedMovies()

        print('Because you like: ', [tag for tag in user_top_tags], '\n')

        print(predictions[['title', 'prediction', 'top_tags']])

    elif recommender_choice == 2:
        print('Creating your personalised recommendations...')

        recommender = CollaborativeFilteringSystem(id)

        recommender.loadModelFromFile('trained_svd.sav')

        predictions = recommender.makePredictions()

        print(predictions[['title', 'prediction', 'top_tags']])
    
    mainMenu(id)


# load system for the active user
def mainMenu(id):
    # print('Booting up the recommender systems for user: ', id, '... \n')
    # time.sleep(2)

    print('Menu: ')
    print('1. View Ratings')
    print('2. Add a New Rating')
    print('3. Edit Rating')
    print('4. Recommendations')
    print('5. Logout')
    print('\n')

    choice = input()

    if (choice) == '1':
        viewRatings(id)
    elif choice == '2':
        addNewRating(id)
    elif choice == '3':
        editRating(id)
    elif choice == '4':
        showRecommendations(id)

# first menu displayed upon run
def startMenu():
    print('--------Movie Recommender System-------- \n')
    print('Welcome to this recommender system. This system saves data about the ratings you and other users have given to films in the database. It also uses tags that users have given to the films, in order to learn details about each film.')
    time.sleep(2)
    print('Menu: ')
    print('1: Login')
    print('2: Create New User')
    print('Please Note: Creating a new user will lead to the collaborative filtering recommender system being offline.')
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

                mainMenu(userID)

            else:
                print('User does not exist in the userbase.')

        # time.sleep(1)

    # creating a new user
    elif choice == '2':
        print('Creating a New User... \n')
        # time.sleep(2)
        number_of_users = 0
        with open('users.json', "r") as users_file:
            info = json.load(users_file)
            number_of_users = len(info['users'])

        newID = number_of_users + 1

        users = []
        with open('users.json', "r") as users_file:
            info = json.load(users_file)

            users = info
            all_users = info['users']
            users = all_users.copy()

            while checkNewID(newID) is not True:
                newID += 1

            users.append(newID)

        to_write = {}
        to_write['users'] = users

        with open('users.json', "w") as users_file:
            to_write = json.dumps(to_write)
            users_file.write(to_write)

        print('Your new User ID is: ', newID)

        # time.sleep(1)

        print('Please remember this so you can login next time!')

# create the users.json file with the database data
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
    startMenu()
    # populateUsers()
