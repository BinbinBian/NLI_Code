__author__ = 'david_torrejon'

from random import seed, shuffle, uniform

def get_test_train_sets(data, percentage_train):
    print ('Data sets generated from data...')
    seed = 1337
    shuffle(data)

    train_set = []
    test_set = []

    for data_point in data:
        num = uniform(1.0, 0.0)
        if num<percentage_train:
            train_set.append(data_point)
        else:
            test_set.append(data_point)

    print('data: %d, train: %d, test: %d')%(len(data), len(train_set), len(test_set))

    return train_set, test_set
