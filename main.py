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

# tracks whether current user has edited database. This is for the collaborative filter
edited_database = False

# used to ensure user input is valid 
def ensure_input_is_integer(input):
    try:
        x = int(input)
        return True
    except ValueError:
        return False

# used to ensure new rating is valid
def ensure_input_is_float(input):
    try:
        x = float(input)
        return True
    except ValueError:
        return False

# returns true if id is unique, and false otherwise. used for creating new user ID
def checkNewID(id):
    with open('users.json', "r") as users_file:
        info = json.load(users_file)
        all_users = info['users']

        if id in all_users:
            return False
        else:
            return True

# returns names of the films for the inputted array of IDs
def getMovieNames(ids):
    names = []
    movie_data = pd.read_csv('./data/movies.csv')

    for id in ids:
        title = movie_data[movie_data['movieId'] == id]['title'].iloc[0]
        names.append(title)

    return names

# let the active user leave another review
def addNewRating(id):
    global edited_database
    movies = pd.read_csv('./data/movies.csv')
    ratings = pd.read_csv('./data/ratings.csv')

    print('Please Note: When you leave a rating for a film, we save this to our database. This data is then used to make better recommendations for you in future.')
    
    valid_movie_id = False 

    # ensure input is valid
    while valid_movie_id is False:
        movie_id = (input('Please enter the ID of the movie you would like to leave a rating for: '))

        if ensure_input_is_integer(movie_id) is False:
            print('Movie ID must be an integer')
        else:
            valid_movie_id = True 
            movie_id = int(movie_id)

    # find movie with inputted ID
    movie = movies.loc[movies['movieId'] == movie_id]

    existing_entry = np.where((ratings['userId'] == int(id)) & (ratings['movieId'] == movie_id))

    # if that movie doesn't exist
    if movie.empty:
        print('No such movie exists with the ID: ', movie_id, '\n')

    # if the movie does exist
    elif len(existing_entry[0]) == 0:
        print('\n')
        print('You are choosing to leave a rating for the film: ', movie['title'].item())
        print('\n')
        print('Please Note: Ratings will be rounded to the nearest 0.5.')

        valid_rating = False
        # ensure rating is valid
        while valid_rating is False:
            rating = input('Please leave a rating for this film (between 0 and 5, ideally in increments of 0.5): ')

            if ensure_input_is_float(rating) is False:
                print('Rating must be a number')
            else:
                rating = float(rating)
                if rating >= 0 and rating <= 5:
                    rating = round(rating * 2) / 2
                    valid_rating = True
                else:
                    print('Invalid Rating. Please try again.')

        # csv file has timestamp column. not used in this recommender system, so different formats in timestamp column of csv file do not matter
        timestamp = datetime.datetime.now().timestamp()

        to_write = {'userId':id, 'movieId':movie_id, 'rating':rating, 'timestamp':timestamp}

        # write new rating to ratings.csv
        with open('./data/ratings.csv', 'a', newline='') as csvfile:
            fieldnames = ['userId','movieId', 'rating', 'timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writerow(to_write)

        print('\n')
        print('Successfully left your review... \n')

        # update variable to reflect changes to data that can't be used in the collaborative filter system
        edited_database = True

    else:
        print('You have already left a review for the film: ', movie['title'].item())
        print('\n')

    mainMenu(id)

# allows active user to edit a previous rating of theirs
def editRating(id):
    print('Editing a previous rating will update your rating in our database, so that we can provide more accurate recommendations in future. \n')
    global edited_database
    ratings = pd.read_csv('./data/ratings.csv')
    movies = pd.read_csv('./data/movies.csv')

    # ensure inputted movie ID is valid
    valid_movie_id = False 
    while valid_movie_id is False:
        movie_id = input('Enter the ID of the movie you would like to edit your rating for: \n')

        if ensure_input_is_integer(movie_id) is False:
            print('Movie ID must be an integer')

        else:
            valid_movie_id = True
            movie_id = int(movie_id)


    rating = ratings.loc[(ratings['userId'] == int(id)) & (ratings['movieId'] == movie_id)]

    movie_info = movies.loc[movies['movieId'] == movie_id]

    # if the movie with inputted ID doesn't exist
    if movie_info.empty:
        print('No such movie exists with the ID: ', movie_id, '\n')

    # if the user hasn't left a previous review of the inputted movie
    elif rating.empty:
        print('You have not left a review for the film: ', movie_info['title'].item(), '\n')

    # edit the previous rating
    else:

        print('Film: ', movie_info['title'].item(), '\n')

        print('Previous Rating: ', rating['rating'].item(), '\n')

        valid_rating = False
        # ensure new rating is valid
        while valid_rating is False:
            new_rating = float(input('Please leave a rating for this film (between 0 and 5, ideally in increments of 0.5): '))
            if new_rating >= 0 and new_rating <= 5:
                new_rating = round(new_rating * 2) / 2
                valid_rating = True
            else:
                print('Invalid Rating. Please try again.')
        print('\n')

        index = np.where((ratings['userId'] == int(id)) & (ratings['movieId'] == movie_id))[0][0]
        # update the ratings dataframe to reflect new rating
        ratings.iloc[index, 2] = new_rating

        print('Updating the rating in the dataset...')

        print('WARNING: This may take some time \n')
        # write the updated dataframe to the ratings.csv file
        ratings.to_csv('./data/ratings.csv', index=False)

        print('\n Successfully updated your rating \n')

        edited_database = True

    mainMenu(id)

# shows active user their past ratings
def viewRatings(id):
    print('Loading your ratings... \n')
    info = pd.read_csv('./data/ratings.csv', index_col=False)

    # find all ratings by the active user
    ratings = (info.loc[info['userId'] == int(id)])

    # if the user has left no ratings
    if ratings.empty:
        print('No ratings to display \n')

    # load all user ratings
    else:

        movie_ids = ratings['movieId'].to_list()

        # print(movie_ids)

        movie_names = getMovieNames(movie_ids)

        result = ratings 

        # drop unnecessary columns to aid with nicer output for user
        result = result.drop('userId', 1)
        result = result.drop('timestamp', 1)
        

        result['movie'] = movie_names

        
        result.set_index('movie', inplace=True)

        # print whole table, not just part of it
        with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
            print(result)

        print('\n')

    mainMenu(id)

# prints recommendations  
def showRecommendations(id):
    # active user can choose between content-based or collaborative filter for their recommendations
    recommender_choice = int(input('1: Content-Based Filter or 2: Collaborative Filter \n'))

    # content-based
    if recommender_choice == 1:
        recommender = ContentBasedSystem(id)

        print('Creating your personalised recommendations...')
        predictions, user_top_tags = recommender.returnPredictedMovies()

        # provide explanations for recommendations
        print('Because you like: ', [tag for tag in user_top_tags], '\n')

        # put prediction scores as percentages
        predictions['prediction'] = predictions['prediction'] * 100

        predictions.rename(columns={'prediction': 'Match (%)'}, inplace=True)

        # round prediction scores to 1 decimal place
        predictions = predictions.round({'Match (%)': 1})

        print(predictions[['title', 'Match (%)', 'top_tags']])

        print('\n')

    # collaborative
    elif recommender_choice == 2:
        # if the active user has updated the ratings.csv file, the model would need to be retrained, which is very time consuming. Thus, this warning is displayed and the model has only been trained off the original dataset.
        if edited_database is True:
            print('WARNING: You have updated the database with either a new rating or editing a previous rating. This updated data will not be reflected in the collaborative filtering recommendations.')
    
        print('Creating your personalised recommendations...')

        recommender = CollaborativeFilteringSystem(id)

        # load pre-trained model, so no need to train model again
        recommender.loadModelFromFile('trained_svd.sav')

        predictions = recommender.makePredictions()

        # put prediction scores as percentages
        predictions['prediction'] = predictions['prediction'] * 100

        predictions.rename(columns={'prediction': 'Match (%)'}, inplace=True)

        # round prediction scores to 1 decimal place
        predictions = predictions.round({'Match (%)': 1})

        print(predictions[['title', 'Match (%)', 'top_tags']])

        print('\n')
    
    mainMenu(id)

# load system for the active user
def mainMenu(id):
    # 5 choices for active user

    print('Menu: ')
    print('1. View Ratings')
    print('2. Add a New Rating')
    print('3. Edit Rating')
    print('4. Recommendations')
    print('5. Logout')
    print('\n')

    valid_option = False 
    # ensure the active user has inputted a valid choice
    while valid_option is False:
        choice = input()

        if choice == '1':
            valid_option = True
            viewRatings(id)
        elif choice == '2':
            valid_option = True
            addNewRating(id)
        elif choice == '3':
            valid_option = True
            editRating(id)
        elif choice == '4':
            valid_option = True
            showRecommendations(id)
        elif choice == '5':
            quit()
        else:
            print('Invalid Option')
            print('Please enter a valid option: ')

# first menu displayed upon run
def startMenu():
    # display welcome message to user on start-up
    print('--------Movie Recommender System-------- \n')
    print('Welcome to this recommender system. This system saves data about the ratings you and other users have given to films in the database. It also uses tags that users have given to the films, in order to learn details about each film.')
    # add delay to aid user experience
    time.sleep(2)

    # display menu options
    print('Menu: ')
    print('1: Login')
    print('2: Create New User')
    print('Please Note: Creating a new user will lead to the collaborative filtering recommender system being offline.')
    print('Please choose an option: \n')

    valid_choice = False
    # ensure the user input is a valid choice from the menu
    while valid_choice is False:
        choice = input()

        # login as an existing user
        if (choice) == '1':
            valid_choice = True
            print('Login \n')

            print('Please enter your User ID: ')

            valid_user_input = False
            # ensure inputted user ID is valid
            while valid_user_input is False:
                userID = input()

                if ensure_input_is_integer(userID) is False:
                    print('User ID must be integer')
                    print('Please enter a valid user ID: ')
                else:
                    valid_user_input = True
                    # check inputted user ID exists in database
                    with open('users.json', "r") as users_file:
                        info = json.load(users_file)
                        existent_users = info['users']

                        # user exists
                        if int(userID) in existent_users:
                            print('\nWelcome User', userID, '\n')

                            mainMenu(userID)

                        # user does not exist
                        else:
                            print('User does not exist in the userbase.')


        # creating a new user
        elif choice == '2':
            valid_choice = True
            print('Creating a New User... \n')

            # check number of users currently in database
            number_of_users = 0
            with open('users.json', "r") as users_file:
                info = json.load(users_file)
                number_of_users = len(info['users'])

            # new user will have a user ID one greater than the last user ID in the database
            newID = number_of_users + 1

            # create array of all users in database, now including this new user
            users = []
            with open('users.json', "r") as users_file:
                info = json.load(users_file)

                all_users = info['users']
                users = all_users.copy()

                while checkNewID(newID) is not True:
                    newID += 1

                users.append(newID)

            # prepare to write the updated users array to the users.json file
            to_write = {}
            to_write['users'] = users

            # update the users.json file
            with open('users.json', "w") as users_file:
                to_write = json.dumps(to_write)
                users_file.write(to_write)

            print('Your new User ID is: ', newID)

            print('Please remember this so you can login next time!')
        
        else:
            print('Invalid Option')
            print('Please enter an available option: ')

# create the users.json file with the database data
# NOTE: this only needs to be ran if users.json does not exist
def populateUsers():
    info = pd.read_csv('./data/ratings.csv', index_col=False)

    # get all user IDs in the ratings.csv file
    users = (info['userId'].unique())

    users = users.tolist()

    # prepare to write users to json file
    to_write = {}
    to_write['users'] = users

    # write all user IDs to users.json
    with open('users.json', "w") as users_file:
        to_write = json.dumps(to_write)
        users_file.write(to_write)


if __name__ == '__main__':
    startMenu()
    # populateUsers()
