"""kNN Classifier."""
import numpy as np


class KNearestNeighbor(object):
    """kNN classifier."""

    def __init__(self):
        pass

    def train(self, X, y):
        """Training function."""
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        """Predict labels for test data using this classifier."""
        dists = self.compute_distances(X)
        return self.predict_labels(dists, k=k)

    def compute_distances(self, X):
        """Compute the distance between each test point in X and each training point."""
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        train_square = np.sum((self.X_train ** 2).T, axis=0)  # a^2
        test_square = np.sum(X ** 2, axis=1)  # b^2
        mid_val = 2 * np.dot(X, self.X_train.T)  # -2ab
        dists = np.sqrt(((-1 * mid_val.T) + test_square).T + train_square)

        return dists

    def predict_labels(self, dists, k=1):
        """Predict labels for samples."""
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = [self.y_train[x] for x in np.argpartition(dists[i, :], k)[:k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))

        return y_pred
