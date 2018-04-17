"""Reduce feautre dimensionality"""

import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


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


def reduce_dimensions(features, clf=None, labels=[]):
    """Reduce the dimensionality of the features using LDA."""
    if not clf:
        clf = LDA(store_covariance=True)
    if len(labels) > 0:
        trans = clf.fit_transform(features, labels)
    else:
        trans = clf.transform(features)
    return trans, clf


def write_to_file(features, files, filename):
    """Write features to a file."""
    f = open(filename, "w")
    for i in range(features.shape[0]):
        f.write(",".join([str(x) for x in features[i]]))
        f.write("," + files[i] + "\n")
    f.close()


if __name__ == "__main__":
    TRAIN_DATA_FOLDER = "/path/to/train/data/images/"
    TEST_DATA_FOLDER = "/path/to/test/data/images/"
    TRAIN_LABLE_FILE = "/path/to/train/data/labels/"
    TEST_LABLE_FILE = "/path/to/test/data/images/"

    TRAIN_FEATURES = "train_word_histograms150.csv"
    TEST_FEATURES = "test_word_histograms150.csv"

    print("Reading labels")
    train_labels = get_labels(TRAIN_DATA_FOLDER, TRAIN_LABLE_FILE)
    test_labels = get_labels(TEST_DATA_FOLDER, TEST_LABLE_FILE)

    print("Reading data")
    X_train, train_files = read_data(TRAIN_FEATURES)
    X_test, test_files = read_data(TEST_FEATURES)
    y_train = np.array([train_labels[x]-1 for x in train_files])
    y_test = np.array([test_labels[x]-1 for x in test_files])

    train_trans, clf = reduce_dimensions(features=X_train, labels=y_train)
    test_trans, clf = reduce_dimensions(features=X_test, labels=y_test, clf=clf)

    write_to_file(train_trans, train_files, TRAIN_FEATURES[0:-4] + "reduced.csv")
    write_to_file(test_trans, test_files, TEST_FEATURES[0:-4] + "reduced.csv")
