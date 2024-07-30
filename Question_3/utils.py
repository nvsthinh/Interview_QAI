import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Computes the Triplet Loss for multiple anchors, positives, and negatives.

    Parameters:
    - anchor: np.ndarray, feature vector of the anchors.
    - positive: np.ndarray, feature vector of the positive.
    - negative: np.ndarray, feature vector of the negative.
    - margin: float, margin for calculating the loss.

    Returns:
    - total_loss: float, the value of the triplet loss.
    """
    # Compute the squared distance between the anchor and the positive example
    pos_dist = np.sum(np.square(anchor - positive), axis=-1)
    
    # Compute the squared distance between the anchor and the negative example
    neg_dist = np.sum(np.square(anchor - negative), axis=-1)
    
    # Compute the Triplet Loss
    loss = np.maximum(0, pos_dist - neg_dist + margin)
    
    return loss

def load_data():
    # Load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
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

def create_triplets(X, y, batch_size):
    """
    Create triplets (anchor, positive, negative) for training.

    Parameters:
    - X: np.ndarray, feature vectors of the samples
    - y: np.ndarray, labels of the samples
    - batch_size: int, number of triplets in the batch

    Returns:
    - anchor: np.ndarray, anchor samples
    - positive: np.ndarray, positive samples
    - negative: np.ndarray, negative samples
    """
    anchor, positive, negative = [], [], []
    for _ in range(batch_size):
        # Select anchor sample
        idx = np.random.randint(0, len(X))
        anchor.append(X[idx])
        
        # Select positive sample (different sample of the same class)
        pos_idxs = np.where(y == y[idx])[0]
        pos_idx = np.random.choice(pos_idxs[pos_idxs != idx])
        positive.append(X[pos_idx])
        
        # Select negative sample (sample of a different class)
        neg_idxs = np.where(y != y[idx])[0]
        neg_idx = np.random.choice(neg_idxs)
        negative.append(X[neg_idx])
    
    return np.array(anchor), np.array(positive), np.array(negative)

def extract_label_features(model, X_train, y_train):
    # Extract one sample for each label
    unique_labels = np.unique(y_train)
    label_samples = {label: X_train[np.where(y_train == label)[0][0]] for label in unique_labels}

    # Extract features for each label sample
    label_features = {label: model.forward(sample) for label, sample in label_samples.items()}

    # Store the features in a list
    label_features_list = [label_features[label] for label in unique_labels]

    return label_features_list, unique_labels

def cosine_similarity(v1, v2):
    """
    Compute the cosine similarity between two vectors.

    Parameters:
    - v1: np.ndarray, first vector
    - v2: np.ndarray, second vector

    Returns:
    - similarity: float, cosine similarity score
    """
    dot_product = np.dot(v1, v2.T)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity
