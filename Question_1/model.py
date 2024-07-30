import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, X, y):
        # Store the number of classes and features
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        # Build the decision tree
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        # Predict labels for input data
        return np.array([self._predict(inputs) for inputs in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        # Stop if maximum depth is reached or only one class is left
        if depth >= self.max_depth or len(set(y)) == 1:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Select random features
        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)
        # Find the best split
        best_feat, best_thresh = self._best_criteria(X, y, feature_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        # Recursively grow the left and right branches
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feature_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        # Loop through features and thresholds to find the best split
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
        # Calculate the entropy of the parent node
        parent_entropy = self._entropy(y)
        # Split the data
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        # Return 0 if no split is possible
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Calculate the weighted entropy of the child nodes
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        # Calculate information gain
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        # Split the data into left and right branches
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        # Calculate the entropy of a label distribution
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        # Return the most common label in the array
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _predict(self, inputs):
        # Predict the label for a single input
        node = self.tree
        while node.left:
            if inputs[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        # Initialize a tree node
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class RandomForest:
    def __init__(self, n_trees=100, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        # Train multiple decision trees
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Predict labels using majority voting from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [Counter(tree_pred).most_common(1)[0][0] for tree_pred in tree_preds]
        return np.array(y_pred)

    def _bootstrap_sample(self, X, y):
        # Generate a bootstrap sample
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def score(self, X, y):
        # Calculate accuracy
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy