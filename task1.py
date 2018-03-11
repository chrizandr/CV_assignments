"""Task 1 of assignment."""

from kNN import KNearestNeighbor
import numpy as np
import pdb
import os


def read_data(filename):
    """Read data from .csv file."""
    data = list()
    files = list()
    f = open(filename, "r")
    for line in f:
        row = line.split(",")
        data.append([float(x) for x in row[0:-1]])
        files.append(row[-1].split('_')[0])
    data = np.array(data)
    return data, files


def get_labels(data_folder, label_file):
    """Get labels for each file."""
    files = os.listdir(data_folder)
    files.sort()
    labels = dict()

    f = open(label_file, "r")
    data = f.read().split(',')
    f.close()

    for i, f in enumerate(files):
        labels[f.split('.')[0]] = int(data[i])

    return labels


if __name__ == "__main__":
    TRAIN_DATA_FOLDER = "/home/chris/cv/hw2_data/train"
    TEST_DATA_FOLDER = "/home/chris/cv/hw2_data/test"
    TRAIN_LABLE_FILE = "/home/chris/cv/hw2_data/train_labels.csv"
    TEST_LABLE_FILE = "/home/chris/cv/hw2_data/test_labels.csv"
    TRAIN_FEATURES = "train_word_histograms.csv"
    TEST_FEATURES = "test_word_histograms.csv"

    print("Reading labels")
    train_labels = get_labels(TRAIN_DATA_FOLDER, TRAIN_LABLE_FILE)
    test_labels = get_labels(TEST_DATA_FOLDER, TEST_LABLE_FILE)

    print("Reading data")
    X_train, train_files = read_data(TRAIN_FEATURES)
    X_test, test_files = read_data(TEST_FEATURES)
    # pdb.set_trace()
    y_train = [train_labels[x]-1 for x in train_files]
    y_test = [test_labels[x]-1 for x in test_files]

    print("Classifying")
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    pred = classifier.predict(X_test, k=5)
    pdb.set_trace()
