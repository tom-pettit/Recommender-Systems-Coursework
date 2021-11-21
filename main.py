import time
import json
from random import randint

# returns true if id is unique, and false otherwise
def checkNewID(id):
    with open('users.json', "r") as users_file:
        info = json.load(users_file)
        all_users = info['users']

        if id in all_users:
            return False
        else:
            return True

def viewRatings(id):
    with open('ratings.json', "r") as ratings_file:
        info = json.load(ratings_file)
        users = info['users']

        if id in users:
            ratings = info['users'][id]['ratings']
            
            print('Your Ratings: \n')
            time.sleep(1)
            for key, value in ratings.items():
                print('Song: ', key, ', Rating: ', value, '\n')

        else:
            print('No Ratings to Date')

def startSystem(id):
    print('Booting up the recommender systems for user: ', id, '... \n')
    time.sleep(2)

    print('Menu: ')
    print('1. View Ratings')
    print('2. Add a New Rating')
    print('3. Logout')
    print('\n')

    choice = input()

    if (choice) == '1':
        viewRatings(id)

def startUI():
    print('--------Song Recommender System--------')
    time.sleep(3)
    print('Menu: ')
    print('1: Login')
    print('2: Create New User')
    print('Please choose an option: \n')

    choice = input()

    # login as an existing user
    if (choice) == '1':
        print('Login \n')
        time.sleep(1)

        print('Please enter your User ID: ')

        userID = input()

        print('Checking if user exists... \n')
        time.sleep(2)
        with open('users.json', "r") as users_file:
            info = json.load(users_file)
            existent_users = info['users']

            if int(userID) in existent_users:
                print('User exists. ')

                startSystem(userID)

            else:
                print('User does not exist in the userbase.')

        time.sleep(1)

    # creating a new user
    elif choice == '2':
        print('Creating a New User... \n')
        time.sleep(2)

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

        time.sleep(1)

        print('Please remember this so you can login next time!')


if __name__ == '__main__':
    startUI()