import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pickle

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Computes the Triplet Loss for given anchor, positive, and negative samples.

    Parameters:
    - anchor: np.ndarray
      Feature vector of the anchor sample.
    - positive: np.ndarray
      Feature vector of the positive sample (same class as anchor).
    - negative: np.ndarray
      Feature vector of the negative sample (different class from anchor).
    - margin: float, default=1.0
      Margin for the loss function to ensure the distance between anchor and negative is greater than the distance 
      between anchor and positive by at least the margin.

    Returns:
    - total_loss: float
      The calculated triplet loss value.
    """
    # Compute the squared distance between the anchor and the positive example
    pos_dist = np.sum(np.square(anchor - positive), axis=-1)
    
    # Compute the squared distance between the anchor and the negative example
    neg_dist = np.sum(np.square(anchor - negative), axis=-1)
    
    # Compute the Triplet Loss using the margin
    loss = np.maximum(0, pos_dist - neg_dist + margin)
    
    return loss

def load_data():
    """
    Load the MNIST dataset, preprocess the data, and split it into training and test sets.

    This function:
    - Fetches the MNIST dataset from OpenML
    - Normalizes the feature values to the range [0, 1]
    - Binarizes the labels (specifically for class 0)
    - Splits the data into training and test sets
    - Reshapes the data to be suitable for MLP models

    Returns:
    - X_train: np.ndarray
      The feature matrix for the training set.
    - X_test: np.ndarray
      The feature matrix for the test set.
    - y_train: np.ndarray
      The labels for the training set.
    - y_test: np.ndarray
      The labels for the test set.
    """
    # Load the MNIST dataset from OpenML
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    
    # Extract features and labels from the dataset
    X = mnist['data'].values
    y = mnist['target'].astype(np.int32).values

    # Normalize the feature values to the range [0, 1]
    X = X / 255.0

    # Binarize the labels (for class 0 vs. non-class 0)
    y = (y == 0).astype(np.int32)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape the data to ensure it's suitable for MLP (N, D) where N is number of samples, D is number of features
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, X_test, y_train, y_test

def create_triplets(X, y, batch_size):
    """
    Generate triplets (anchor, positive, negative) for training.

    Parameters:
    - X: np.ndarray
      Feature vectors of the samples.
    - y: np.ndarray
      Labels of the samples.
    - batch_size: int
      Number of triplets to generate.

    Returns:
    - anchor: np.ndarray
      Array of anchor samples.
    - positive: np.ndarray
      Array of positive samples corresponding to anchors.
    - negative: np.ndarray
      Array of negative samples corresponding to anchors.
    """
    anchor, positive, negative = [], [], []
    for _ in range(batch_size):
        # Select a random anchor sample
        idx = np.random.randint(0, len(X))
        anchor.append(X[idx])
        
        # Select a positive sample (different sample of the same class as the anchor)
        pos_idxs = np.where(y == y[idx])[0]
        pos_idx = np.random.choice(pos_idxs[pos_idxs != idx])
        positive.append(X[pos_idx])
        
        # Select a negative sample (sample of a different class from the anchor)
        neg_idxs = np.where(y != y[idx])[0]
        neg_idx = np.random.choice(neg_idxs)
        negative.append(X[neg_idx])
    
    return np.array(anchor), np.array(positive), np.array(negative)

def extract_label_features(model, X_train, y_train):
    """
    Extract feature vectors for one sample of each label from the training set using the provided model.

    Parameters:
    - model: object
      The model used to extract features.
    - X_train: np.ndarray
      Feature matrix of the training set.
    - y_train: np.ndarray
      Labels of the training set.

    Returns:
    - label_features_list: list of np.ndarray
      List of feature vectors, one for each unique label.
    - unique_labels: np.ndarray
      Array of unique labels.
    """
    # Extract one sample for each label
    unique_labels = np.unique(y_train)
    label_samples = {label: X_train[np.where(y_train == label)[0][0]] for label in unique_labels}

    # Extract features for each label sample using the model
    label_features = {label: model.forward(sample) for label, sample in label_samples.items()}

    # Store the features in a list
    label_features_list = [label_features[label] for label in unique_labels]

    return label_features_list, unique_labels

def cosine_similarity(v1, v2):
    """
    Compute the cosine similarity between two vectors.

    Parameters:
    - v1: np.ndarray
      The first vector.
    - v2: np.ndarray
      The second vector.

    Returns:
    - similarity: float
      The cosine similarity score between the two vectors.
    """
    dot_product = np.dot(v1, v2.T)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity

def load_pickle(filename):
    with open(filename, 'rb') as f:
        variable =  pickle.load(f)
    return variable