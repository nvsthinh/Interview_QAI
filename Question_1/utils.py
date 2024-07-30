import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_data():
    # Load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist['data'].values
    y = mnist['target'].astype(np.int32).values

    # Binarize the labels (convert to binary classification for simplicity)
    y = (y == 0).astype(np.int32)  # Let's classify '0' vs 'not 0'

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test