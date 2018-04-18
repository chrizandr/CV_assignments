"""Task 2 of assignment."""

from SVM import SVM
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import pdb


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
    TRAIN_DATA_FOLDER = "/home/chrizandr/CV_assignments/data/hw2_data/train"
    TEST_DATA_FOLDER = "/home/chrizandr/CV_assignments/data/hw2_data/test"
    TRAIN_LABLE_FILE = "/home/chrizandr/CV_assignments/data/hw2_data/train_labels.csv"
    TEST_LABLE_FILE = "/home/chrizandr/CV_assignments/data/hw2_data/test_labels.csv"
    TRAIN_FEATURES = "train_word_histograms200.csv"
    TEST_FEATURES = "test_word_histograms200.csv"

    print("Reading labels")
    train_labels = get_labels(TRAIN_DATA_FOLDER, TRAIN_LABLE_FILE)
    test_labels = get_labels(TEST_DATA_FOLDER, TEST_LABLE_FILE)

    print("Reading data")
    X_train, train_files = read_data(TRAIN_FEATURES)
    X_test, test_files = read_data(TEST_FEATURES)
    # pdb.set_trace()
    y_train = np.array([train_labels[x]-1 for x in train_files])
    y_test = np.array([test_labels[x]-1 for x in test_files])

    best_acc = -1
    best_params = (None, None)
    for i in range(10):
        print("10 Fold Crossval , round {}".format(i))
        shuffle_indices = np.random.permutation(X_train.shape[0])
        X_train_val = X_train[shuffle_indices]
        y_train_val = y_train[shuffle_indices]

        val_split = int(X_train.shape[0] * 0.9)
        X_val = X_train_val[val_split:]
        y_val = y_train_val[val_split:]

        X_train_val = X_train_val[0:val_split]
        y_train_val = y_train_val[0:val_split]

        lis1 = []
        for l in [1e-3, 1e-2, 1e-1]:
            lis2 = []
            for r in [1e-5, 1e-4, 1e-3]:
                classifier = SVM()
                classifier.train(X_train_val, y_train_val, learning_rate=l, reg=r)
                pred = classifier.predict(X_val)
                acc = (pred == y_val).sum() * 100 / pred.shape[0]
                print("Accuracy {}%".format(acc))
                lis2.append(acc)
                if acc > best_acc:
                    acc = best_acc
                    best_params = (l, r)
            lis1.append(lis2)
    best_l, best_r = best_params
    print("Best params are l={}, r={}".format(best_l, best_r))

    print("Classifying")
    shuffle_indices = np.random.permutation(X_train.shape[0])
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    classifier = SVM()
    classifier.train(X_train, y_train, learning_rate=best_l, reg=best_r)
    pred = classifier.predict(X_test)
    acc = (pred == y_test).sum() * 100 / pred.shape[0]
    print("Accuracy of model {}%".format(acc))
    cm = confusion_matrix(pred, y_test)
    print(cm)
