import numpy as np
from collections import Counter

class DecisionTree:
    """
    A Decision Tree classifier.

    Parameters:
    - max_depth: int, optional (default=10)
      The maximum depth of the tree.

    Attributes:
    - n_classes: int
      The number of unique classes in the training data.
    - n_features: int
      The number of features in the training data.
    - tree: Node
      The root node of the decision tree.
    """

    def __init__(self, max_depth):
        """
        Initialize the Decision Tree with a maximum depth.

        Parameters:
        - max_depth: int
          The maximum depth of the tree.
        """
        self.max_depth = max_depth

    def fit(self, X, y):
        """
        Fit the decision tree to the training data.

        Parameters:
        - X: numpy.ndarray
          The feature matrix of shape (n_samples, n_features).
        - y: numpy.ndarray
          The target vector of shape (n_samples,).
        """
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        """
        Predict the labels for the input data.

        Parameters:
        - X: numpy.ndarray
          The feature matrix of shape (n_samples, n_features).

        Returns:
        - numpy.ndarray
          The predicted labels for each sample in X.
        """
        return np.array([self._predict(inputs) for inputs in X])

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.

        Parameters:
        - X: numpy.ndarray
          The feature matrix.
        - y: numpy.ndarray
          The target vector.
        - depth: int, optional (default=0)
          The current depth of the tree.

        Returns:
        - Node
          The root node of the built subtree.
        """
        n_samples, n_features = X.shape
        if depth >= self.max_depth or len(set(y)) == 1:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_criteria(X, y, feature_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feature_idxs):
        """
        Find the best feature and threshold to split the data.

        Parameters:
        - X: numpy.ndarray
          The feature matrix.
        - y: numpy.ndarray
          The target vector.
        - feature_idxs: numpy.ndarray
          The indices of features to consider for splitting.

        Returns:
        - split_idx: int
          The index of the best feature to split on.
        - split_thresh: float
          The threshold value for the best split.
        """
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feature_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        """
        Calculate the information gain of a split.

        Parameters:
        - y: numpy.ndarray
          The target vector.
        - X_column: numpy.ndarray
          The feature column to split on.
        - split_thresh: float
          The threshold value for the split.

        Returns:
        - ig: float
          The information gain of the split.
        """
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        """
        Split the data into left and right branches based on a threshold.

        Parameters:
        - X_column: numpy.ndarray
          The feature column to split on.
        - split_thresh: float
          The threshold value for the split.

        Returns:
        - left_idxs: numpy.ndarray
          Indices of samples in the left branch.
        - right_idxs: numpy.ndarray
          Indices of samples in the right branch.
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        """
        Calculate the entropy of a label distribution.

        Parameters:
        - y: numpy.ndarray
          The target vector.

        Returns:
        - entropy: float
          The entropy of the label distribution.
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        """
        Return the most common label in the array.

        Parameters:
        - y: numpy.ndarray
          The target vector.

        Returns:
        - most_common: int
          The most common label.
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _predict(self, inputs):
        """
        Predict the label for a single input.

        Parameters:
        - inputs: numpy.ndarray
          The feature vector for a single sample.

        Returns:
        - label: int
          The predicted label.
        """
        node = self.tree
        while node.left:
            if inputs[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

class Node:
    """
    A node in the decision tree.

    Parameters:
    - feature: int, optional
      The index of the feature used for the split.
    - threshold: float, optional
      The threshold value for the split.
    - left: Node, optional
      The left child node.
    - right: Node, optional
      The right child node.
    - value: int, optional
      The label value for leaf nodes.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        Initialize a tree node.

        Parameters:
        - feature: int, optional
          The index of the feature used for the split.
        - threshold: float, optional
          The threshold value for the split.
        - left: Node, optional
          The left child node.
        - right: Node, optional
          The right child node.
        - value: int, optional
          The label value for leaf nodes.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class RandomForest:
    """
    A Random Forest classifier.

    Parameters:
    - n_trees: int, optional (default=100)
      The number of trees in the forest.
    - max_depth: int, optional (default=10)
      The maximum depth of each tree.

    Attributes:
    - trees: list
      A list of trained Decision Trees.
    """

    def __init__(self, n_trees=100, max_depth=10):
        """
        Initialize the Random Forest with the number of trees and maximum depth.

        Parameters:
        - n_trees: int
          The number of trees in the forest.
        - max_depth: int
          The maximum depth of each tree.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """
        Train multiple decision trees using bootstrap sampling.

        Parameters:
        - X: numpy.ndarray
          The feature matrix of shape (n_samples, n_features).
        - y: numpy.ndarray
          The target vector of shape (n_samples,).
        """
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict the labels using majority voting from all trees.

        Parameters:
        - X: numpy.ndarray
          The feature matrix of shape (n_samples, n_features).

        Returns:
        - numpy.ndarray
          The predicted labels for each sample in X.
        """
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [Counter(tree_pred).most_common(1)[0][0] for tree_pred in tree_preds]
        return np.array(y_pred)

    def _bootstrap_sample(self, X, y):
        """
        Generate a bootstrap sample from the data.

        Parameters:
        - X: numpy.ndarray
          The feature matrix.
        - y: numpy.ndarray
          The target vector.

        Returns:
        - X_sample: numpy.ndarray
          The bootstrap sample of features.
        - y_sample: numpy.ndarray
          The bootstrap sample of targets.
        """
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def score(self, X, y):
        """
        Calculate the accuracy of the predictions.

        Parameters:
        - X: numpy.ndarray
          The feature matrix of shape (n_samples, n_features).
        - y: numpy.ndarray
          The true labels of shape (n_samples,).

        Returns:
        - accuracy: float
          The accuracy of the predictions.
        """
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy