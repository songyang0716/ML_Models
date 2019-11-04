# Gradient boosting tree
import sys
sys.path.append('/Users/yangsong/Desktop/Projects/ml_algorithm_code/tree')

import numpy as np
from decision_tree import DecisionTree
from losses import MSELoss, CrossEntropyLoss


def to_one_hot(labels, n_classes=None):
    if labels.ndim > 1:
        raise ValueError(
            "labels must have dimension 1, but got {}".format(labels.ndim))
    N = labels.size
    n_cols = np.max(labels) + 1 if n_classes is None else n_classes
    one_hot = np.zeros((N, n_cols))
    one_hot[np.arange(N), labels] = 1.0
    return one_hot


class GradientBoostedDecisionTree:
    """
    An instance of gradient boosted machines (GBM) using decision trees as the
    weak learners.
    GBMs fit an ensemble of m weak learners s.t.:
        f_m(X) = b(X) + lr * w_1 * g_1 + ... + lr * w_m * g_m
    where b is a fixed initial estimate for the targets, lr is a learning rate
    parameter, and w* and g* denote the weights and learner predictions for
    subsequent fits.
    We fit each w and g iteratively using a greedy strategy so that at each
    iteration i,
        w_i, g_i = argmin_{w_i, g_i} L(Y, f_{i-1}(X) + w_i * g_i)
    On each iteration we fit a new weak learner to predict the negative
    gradient of the loss with respect to the previous prediction, f_{i-1}(X).
    We then use the element-wise product of the predictions of this weak
    learner, g_i, with a weight, w_i, to compute the amount to adjust the
    predictions of our model at the previous iteration, f_{i-1}(X):
        f_i(X) := f_{i-1}(X) + w_i * g_i
    """

    def __init__(
        self,
        n_iter,
        max_depth=None,
        classifier=True,
        learning_rate=1,
        loss="crossentropy",
        step_size="constant",
        split_criterion="entropy",
    ):
        """
        A gradient boosted ensemble of decision trees.
        Parameters
        ----------
        n_iter : int
            The number of iterations / weak estimators to use when fitting each
            dimension/class of Y
        max_depth : int (default: None)
            The maximum depth of each decision tree weak estimator
        classifier : bool (default: True)
            Whether Y contains class labels or real-valued targets
        learning_rate : float (default: 1)
            Value in [0, 1] controlling the amount each weak estimator
            contributes to the overall model prediction. Sometimes known as the
            `shrinkage parameter` in the GBM literature
        loss : str
            The loss to optimize for the GBM. Valid entries are {"crossentroy",
            "mse"}.
        step_size : str
            How to choose the weight for each weak learner. Valid entries are
            {"constant", "adaptive"}. If "constant", use a fixed weight of 1
            for each learner. If "adaptive", use a step size computed via
            line-search on the current iteration's loss.
            In this example, we only used fixed step size 1
        split_criterion : str
            The error criterion to use when calculating splits for each weak
            learner. When `classifier` is False, valid entries are {'mse'}.
            When `classifier` is True, valid entries are {'entropy', 'gini'}.
        """
        self.loss = loss
        self.weights = None
        self.learners = None
        self.out_dims = None
        self.n_iter = n_iter
        self.base_estimator = None
        self.max_depth = max_depth
        self.step_size = step_size
        self.classifier = classifier
        self.learning_rate = learning_rate
        self.split_criterion = split_criterion

    def fit(self, X, Y):
        if self.loss == "mse":
            loss = MSELoss()
        elif self.loss == "crossentropy":
            loss = CrossEntropyLoss()

        # convert Y to one-hot encoding
        if self.classifier:
            Y = to_one_hot(Y.flatten())
        else:
            # if the shape of Y is like (N,), then we need to convert it to be (N,1)
            Y = Y.reshape(-1, 1) if len(Y.shape) == 1 else Y

        N, M = X.shape
        # out_dims is the number of classes in the outcome. 1 for continuous Y
        self.out_dims = Y.shape[1]
        # Record all the learners (all the trees)
        self.learners = np.empty((self.n_iter, self.out_dims), dtype=object)
        # weights for each prediction (since we don't do the linear search step, we just put the learning_rate here)
        # The very first iteration has weights equals to 1 (so we don't multiply self.learning_rate)
        self.weights = np.ones((self.n_iter, self.out_dims))
        self.weights[1:, :] *= self.learning_rate

        # Prediction values, N samples, and each samples has self.out_dims dimensions
        Y_pred = np.zeros((N, self.out_dims))
        # Very first iteration, use mean to predict
        for k in range(self.out_dims):
            t = loss.base_estimator()
            t.fit(X, Y[:, k])
            Y_pred[:, k] = t.predict(X)
            self.learners[0, k] = t

        # Incrementally fit each learner on the negative gradient of the loss
        # wrt the previous fit (pseudo-residuals)
        for i in range(1, self.n_iter):
            for k in range(self.out_dims):
                y, y_pred = Y[:, k], Y_pred[:, k]
                neg_grad = -1 * loss.grad(y, y_pred)
                # use MSE as the surrogate loss when fitting to negative gradients
                t = DecisionTree(
                    classifier=False, max_depth=self.max_depth, criterion="mse"
                )
                # fit X to negative gradients of the current loss function
                t.fit(X, neg_grad)
                self.learners[i, k] = t

                # compute step size and weight for the current learner
                step = 1.0
                h_pred = t.predict(X)

                # We ignore the linear search step

                # update weights and our overall prediction for Y
                self.weights[i, k] *= step
                Y_pred[:, k] += self.weights[i, k] * h_pred

    def predict(self, X):
        Y_pred = np.zeros((X.shape[0], self.out_dims))
        for i in range(1, self.n_iter):
            for k in range(self.out_dims):
                Y_pred[:, k] += (self.learners[i,k].predict(X)) * self.weights[i, k]

        if self.classifier:
            Y_pred = np.argmax(Y_pred, axis=1)
        return Y_pred