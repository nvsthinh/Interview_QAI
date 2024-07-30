import numpy as np

class DecisionStump:
    def __init__(self):
        # Initialize the stump's parameters
        self.polarity = 1      # Polarity of the stump (+1 or -1)
        self.feature_index = None   # Index of the feature to split on
        self.threshold = None      # Threshold for the feature split
        self.alpha = None      # Weight of the stump in the final classifier

    def predict(self, X):
        """
        Predict the labels for the given samples.

        Parameters:
        - X: numpy array of shape (n_samples, n_features), the input features

        Returns:
        - predictions: numpy array of shape (n_samples,), the predicted labels
        """
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]  # Extract the feature column for prediction
        predictions = np.ones(n_samples)      # Initialize predictions to 1 (default class)
        
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1  # Set to -1 if below threshold
        else:
            predictions[X_column > self.threshold] = -1  # Set to -1 if above threshold
        
        return predictions

class AdaBoost:
    def __init__(self, n_clf=5):
        """
        Initialize the AdaBoost classifier.

        Parameters:
        - n_clf: int, number of weak classifiers (decision stumps)
        """
        self.n_clf = n_clf  # Number of classifiers (stumps) to use

    def fit(self, X, y):
        """
        Fit the AdaBoost model.

        Parameters:
        - X: numpy array of shape (n_samples, n_features), the input features
        - y: numpy array of shape (n_samples,), the target labels
        """
        n_samples, n_features = X.shape
        y = np.where(y == 0, -1, 1)  # Convert labels from {0, 1} to {-1, 1}

        # Initialize weights for all samples
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []  # List to store the weak classifiers

        for _ in range(self.n_clf):
            clf = DecisionStump()  # Create a new decision stump
            min_error = float('inf')  # Initialize the minimum error to infinity

            # Loop over all features
            for feature_i in range(n_features):
                X_column = X[:, feature_i]  # Extract feature column
                thresholds = np.unique(X_column)  # Get unique values as thresholds

                for threshold in thresholds:
                    p = 1  # Initialize polarity to +1
                    predictions = np.ones(n_samples)  # Initialize predictions to +1
                    predictions[X_column < threshold] = -1  # Predictions for the current threshold

                    # Calculate the weighted error for the current stump
                    error = np.sum(w * (predictions != y))

                    # If the error is greater than 50%, flip the polarity
                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # Update the best stump if this one has lower error
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error

            # Calculate the weight (alpha) of the stump
            clf.alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-10))

            # Update weights for the next round
            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y * predictions)  # Adjust weights based on error
            w /= np.sum(w)  # Normalize weights

            self.clfs.append(clf)  # Store the trained stump

    def predict(self, X):
        """
        Predict the labels for the given samples.

        Parameters:
        - X: numpy array of shape (n_samples, n_features), the input features

        Returns:
        - y_pred: numpy array of shape (n_samples,), the predicted labels
        """
        clf_preds = np.zeros(X.shape[0])  # Initialize predictions to zero

        # Sum the weighted predictions from all classifiers
        for clf in self.clfs:
            clf_preds += clf.alpha * clf.predict(X)

        # Determine final class based on the sign of the aggregated predictions
        y_pred = np.sign(clf_preds)
        return np.where(y_pred == -1, 0, 1)  # Convert {-1, 1} to {0, 1}