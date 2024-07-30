import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Tính toán Triplet Loss.

    Parameters:
    - anchor: np.ndarray, vector đặc trưng của anchor.
    - positive: np.ndarray, vector đặc trưng của positive.
    - negative: np.ndarray, vector đặc trưng của negative.
    - margin: float, margin để tính toán loss.

    Returns:
    - loss: float, giá trị của triplet loss.
    """
    # Tính toán khoảng cách bình phương giữa anchor và positive
    pos_dist = np.sum(np.square(anchor - positive), axis=-1)
    
    # Tính toán khoảng cách bình phương giữa anchor và negative
    neg_dist = np.sum(np.square(anchor - negative), axis=-1)
    
    # Tính toán Triplet Loss
    loss = np.maximum(0, pos_dist - neg_dist + margin)
    
    return loss

def load_data():
    # Load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist['data'].values
    y = mnist['target'].astype(np.int32).values

    # Normalize the data
    X = X / 255.0

    # Binarize the labels
    y = (y == 0).astype(np.int32)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure the data is in the right shape for MLP (N, D) where N is number of samples, D is number of features
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, X_test, y_train, y_test