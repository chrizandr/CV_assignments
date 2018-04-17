"""Use kNN"""

from kNN import KNearestNeighbor
import numpy as np
import pdb
import os
from sklearn.metrics import confusion_matrix


def read_data(filename):
    """Read data from .csv file."""
    data = list()
    files = list()
    f = open(filename, "r")
    for line in f:
        row = line.strip().split(",")
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
    TRAIN_DATA_FOLDER = "/path/to/train/data/images/"
    TEST_DATA_FOLDER = "/path/to/test/data/images/"
    TRAIN_LABLE_FILE = "/path/to/train/data/labels/"
    TEST_LABLE_FILE = "/path/to/test/data/images/"

    TRAIN_FEATURES = "train_word_histograms150reduced.csv"
    TEST_FEATURES = "test_word_histograms150reduced.csv"

    print("Reading labels")
    train_labels = get_labels(TRAIN_DATA_FOLDER, TRAIN_LABLE_FILE)
    test_labels = get_labels(TEST_DATA_FOLDER, TEST_LABLE_FILE)

    print("Reading data")
    X_train, train_files = read_data(TRAIN_FEATURES)
    X_test, test_files = read_data(TEST_FEATURES)
    # pdb.set_trace()
    y_train = np.array([train_labels[x]-1 for x in train_files])
    y_test = np.array([test_labels[x]-1 for x in test_files])

    print("Classifying")
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    for k in range(1, 7):
        pred = classifier.predict(X_test, k=k)
        accuracy = float((pred == y_test).sum()) * 100 / pred.shape[0]
        print("Accuracy for k = {} is {}%".format(k, accuracy))
    cm = confusion_matrix(pred, y_test)
    print(cm)
