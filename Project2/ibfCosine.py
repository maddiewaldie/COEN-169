import math

# Initialize user_data dictionary to store user ratings
user_data = {}

# Read data from the training set
try:
    with open("train.txt", "r") as file:
        train_users = set()  # Set to store unique user IDs in the training set
        for line in file:
            u, m, r = [int(i) for i in line.split()]  # Extract user ID, movie ID, and rating
            if u not in train_users:
                user_data[u] = {}  # Create a dictionary for each user
            train_users.add(u)
            user_data[u][m] = r  # Store the rating for the user and movie
except FileNotFoundError:
    print("error: File train.txt not found")

# Read data from test20.txt
try:
    with open("test5.txt", "r") as file2:
        test_users = set()  # Set to store unique user IDs in the test set
        for line in file2:
            u, m, r = [int(i) for i in line.split()]  # Extract user ID, movie ID, and rating
            if u not in test_users:
                user_data[u] = {}  # Create a dictionary for each test user
            test_users.add(u)
            user_data[u][m] = r  # Store the rating for the test user and movie
except FileNotFoundError:
    print("error: File test5.txt not found")

# Iterate through test_users and train_users to find shared movies
similarities = {}
testmean = 0
testcount = 0
trainmean = 0
traincount = 0
numRaters = 0  # Number of users that have rated a particular movie
totalusers = len(test_users) + len(train_users)

for test_user in test_users:
    # Set up list of similarities for each test user
    similarities[test_user] = []

    for train_user in train_users:
        test_movies = user_data[test_user].keys()
        train = []  # List of train_user's ratings given to shared movies
        test = []  # List of test_user's ratings given to shared movies
        similarityUsers = set()
        similarityUsers.add(test_user)

        # The ratings in train[] and test[] correspond to the same movie
        for m in test_movies:
            if user_data[test_user][m] == 0:  # Movie not rated
                continue
            if m in user_data[train_user]:
                train.append(user_data[train_user][m])
                test.append(user_data[test_user][m])
                trainmean += user_data[train_user][m]
                traincount += 1
                testmean += user_data[test_user][m]
                testcount += 1
                numRaters += 1

        trainmean = trainmean / traincount
        testmean = testmean / testcount
        numerator = 0

        # Cosine similarity calculation
        dot_product = sum(train[i] * test[i] for i in range(len(test)))
        test_mag = math.sqrt(sum(test[i] ** 2 for i in range(len(test))))
        train_mag = math.sqrt(sum(train[i] ** 2 for i in range(len(train))))

        similarity = dot_product / (test_mag * train_mag) if (test_mag * train_mag) != 0 else 0

        # Append the calculated similarity and train_user to the similarities list
        similarities[test_user].append((similarity, train_user))

    # Sort similarities in descending order
    similarities[test_user].sort(reverse=True)

k = 35  # Set the number of similar users to consider

output = []
for test_user in test_users:
    test_movies = user_data[test_user].keys()
    for m in test_movies:
        if user_data[test_user][m] != 0:
            continue
        count = 0
        train_ratings = []
        train_similarities = []

        # Iterate through similarities for each test user and gather ratings and similarities
        for x in similarities[test_user]:
            similarity = x[0]
            train_user = x[1]

            # Skip over training users that did not rate the target movie
            if m not in user_data[train_user]:
                continue

            train_ratings.append(user_data[train_user][m])
            train_similarities.append(similarity)
            count += 1

            if count == k:  # Limit the number of similar users considered
                break

        numerator = sum(train_ratings[i] * train_similarities[i] for i in range(len(train_ratings)))
        denominator = sum(abs(train_similarities[i]) for i in range(len(train_similarities)))

        # Calculate the final predicted rating
        rating = numerator / denominator if denominator != 0 else 0

        # Ensure the rating is between 1 and 5
        rating = max(1, min(5, round(rating)))

        # Save results to the output list
        output.append(f"{test_user} {m} {rating}\n")

# Write output to a file
with open("result5.txt", "w") as file:
    file.writelines(output)
