"""Create Bag of Visual words."""
import os
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans as KMeans
import collections


def read_features(folder, split=False, return_files=False):
    """Read features from all images."""
    files = os.listdir(folder)
    data = []
    for f in files:
        # print("Processing file", f)
        file_ = open(folder + f, "r")
        if split:
            img_data = []
        for line in file_:
            if split:
                img_data.append([float(x) for x in line.split(',')[4:]])
            else:
                data.append([float(x) for x in line.split(',')[4:]])
        if split:
            data.append(img_data)
    if split:
        if return_files:
            return data, files
        return data

    data = np.array(data)
    if return_files:
        return data, files

    return data


def cluster_words(X, n_clusters, model_name=None):
    """Cluster strokes, calculate distribution, group images according to clusters."""
    kmeans = KMeans(n_clusters=n_clusters, init='random', max_iter=20000, batch_size=50000, init_size=10000)

    print("Clustering the words")
    kmeans.fit(X)

    if not model_name:
        model_name = "visual_words_" + str(n_clusters) + ".pkl"

    print("Saving model as", model_name)
    pickle.dump(kmeans, open(model_name, "wb"))
    return model_name


def create_word_histogram(word_cluster_file, X, normalize=True, scale=True):
    """Create histogram of visual words for each image."""
    histograms = []
    cluster = pickle.load(open(word_cluster_file, "rb"))
    words = np.sort(np.unique(cluster.labels_))

    for img in X:
        predictions = cluster.predict(img)
        counter = collections.Counter(predictions)
        histograms.append([counter[x] if x in counter else 0 for x in words])
    histograms = np.array(histograms)

    if normalize:
        sum_hist = np.sum(histograms, axis=1)
        histograms = histograms / sum_hist[:, None]
    if scale:
        histograms = histograms * 100

    return histograms


def train_visual_words(TRAIN_FOLDERS, TEST_FOLDERS, n_clusters):
    """Create visual word histograms of size n_clusters."""
    print("Training visual words.")
    combined_data = read_features(TRAIN_FOLDERS)
    model_name = cluster_words(combined_data, n_clusters)

    print("Creating histograms of words for train images")
    split_data, files = read_features(TRAIN_FOLDERS, split=True, return_files=True)
    histograms = create_word_histogram(model_name, split_data, n_clusters)
    write_to_file(histograms, files, "train_word_histograms.csv")

    print("Creating histograms of words for test images")
    split_data, files = read_features(TEST_FOLDERS, split=True, return_files=True)
    histograms = create_word_histogram(model_name, split_data, n_clusters)
    write_to_file(histograms, files, "test_word_histograms.csv")

    return True


def write_to_file(histograms, files, filename):
    f = open(filename, "w")
    for i in range(histograms.shape[0]):
        f.write(",".join([str(x) for x in histograms[i]]))
        f.write("," + files[i] + "\n")
    f.close()


if __name__ == "__main__":
    TEST_FOLDERS = "/home/chris/cv/hw2_data/test_sift_features/"
    TRAIN_FOLDERS = "/home/chris/cv/hw2_data/train_sift_features/"
    N_WORDS = 100

    assert train_visual_words(TRAIN_FOLDERS, TEST_FOLDERS, N_WORDS)
