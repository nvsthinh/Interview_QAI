import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load the MNIST dataset and split it into training and test sets.

    The MNIST dataset consists of 28x28 pixel images of handwritten digits (0-9) and their corresponding labels.
    This function fetches the dataset from OpenML, extracts the features and labels, and splits the data into
    training and test sets.

    Returns:
    - X_train: numpy.ndarray
      The feature matrix for the training set.
    - X_test: numpy.ndarray
      The feature matrix for the test set.
    - y_train: numpy.ndarray
      The labels for the training set.
    - y_test: numpy.ndarray
      The labels for the test set.
    """

    # Load the MNIST dataset from OpenML
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    
    # Extract features and labels from the dataset
    X = mnist['data'].values
    y = mnist['target'].astype(np.int32).values

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test