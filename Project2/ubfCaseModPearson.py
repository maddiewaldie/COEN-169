import math

# Initialize user_data dictionary to store user ratings
user_data = {}

# Read data from the training set
try:
    with open("train.txt", "r") as file:
        train_users = set()
        for line in file:
            u, m, r = [int(i) for i in line.split()]
            if u not in train_users:
                user_data[u] = {}
            train_users.add(u)
            user_data[u][m] = r
except FileNotFoundError:
    print("error: File train.txt not found")

# Read data from test20.txt
try:
    with open("test5.txt", "r") as file2:
        test_users = set()
        for line in file2:
            u, m, r = [int(i) for i in line.split()]
            if u not in test_users:
                user_data[u] = {}
            test_users.add(u)
            user_data[u][m] = r
except FileNotFoundError:
    print("error: File test5.txt not found")

# Iterate through test_users and train_users to find shared movies
similarities = {}
testmean = 0
testcount = 0
trainmean = 0
traincount = 0
numRaters = 0
totalusers = len(test_users) + len(train_users)

for test_user in test_users:
    similarities[test_user] = []

    for train_user in train_users:
        test_movies = user_data[test_user].keys()
        train = []
        test = []
        similarityUsers = set()
        similarityUsers.add(test_user)

        for m in test_movies:
            if user_data[test_user][m] == 0:
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

        for i in range(len(test)):
            numerator += (train[i] - trainmean) * (test[i] - testmean)

        test_mag = math.sqrt(sum((test[i] - testmean) ** 2 for i in range(len(test))))
        train_mag = math.sqrt(sum((train[i] - trainmean) ** 2 for i in range(len(train))))

        denominator = test_mag * train_mag

        if denominator == 0:
            prediction = 0
        else:
            prediction = (numerator / denominator)

        # Case modification: Consider user and item biases
        prediction = prediction + (testmean + trainmean) / 2

        similarities[test_user].append((prediction, train_user))

    similarities[test_user].sort(reverse=True)

k = 15

output = []
for test_user in test_users:
    test_movies = user_data[test_user].keys()
    for m in test_movies:
        if user_data[test_user][m] != 0:
            continue
        count = 0
        train_ratings = []
        train_similarities = []

        for x in similarities[test_user]:
            prediction, train_user = x

            if m not in user_data[train_user]:
                continue

            train_ratings.append(user_data[train_user][m])
            train_similarities.append(prediction)
            count += 1

            if count == k:
                break

        numerator = sum(train_ratings[i] * train_similarities[i] for i in range(len(train_ratings)))
        denominator = sum(abs(train_similarities[i]) for i in range(len(train_similarities)))

        rating = numerator / denominator if denominator != 0 else 0

        rating = max(1, min(5, round(rating)))

        output.append(f"{test_user} {m} {rating}\n")

with open("result5_modified.txt", "w") as file:
    file.writelines(output)
