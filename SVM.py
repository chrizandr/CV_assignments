"""Code for SVM."""
import numpy as np


class SVM(object):
    """SVM Classifier."""
    def __init__(self):
        self.W = None

    def loss(self, X, y, reg):
        """SVM loss function."""

        loss = 0.0
        dW = np.zeros(self.W.shape)  # initialize the gradient as zero
        scores = X.dot(self.W)
        num_train = X.shape[0]

        correct_class_score = scores[np.arange(num_train), y].reshape(num_train, 1)
        margin = scores - correct_class_score + 1

        margins = np.maximum(np.zeros(scores.shape), margin)
        margins[np.arange(num_train), y] = 0

        loss = margins.sum()/num_train
        loss += 0.5 * reg * np.sum(self.W * self.W)

        mid = np.zeros(margins.shape)
        mid[margins > 0] = 1
        mid[range(num_train), list(y)] = 0
        k_count = -1 * np.sum(mid, axis=1)
        mid[range(num_train), list(y)] = k_count

        dW = (X.T).dot(mid)
        dW = dW/num_train + reg*self.W

        return loss, dW

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """Training function."""
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            print("Training >>>> Step {}".format(str(it)))
            self.W += -1 * learning_rate * grad

        return loss_history

    def predict(self, X):
        """Predict labels."""
        y_pred = np.zeros(X.shape[1])

        scores = np.dot(X, self.W)
        y_pred = np.argmax(scores, axis=1)

        return y_pred
