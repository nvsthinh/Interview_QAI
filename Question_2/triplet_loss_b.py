import numpy as np

def triplet_loss_multi_samples(anchors, positives, negatives, margin=1.0):
    """
    Computes the Triplet Loss for multiple anchors, positives, and negatives.

    Parameters:
    - anchors: list of np.ndarray, list of feature vectors of the anchors.
    - positives: list of np.ndarray, list of feature vectors of the positives.
    - negatives: list of np.ndarray, list of feature vectors of the negatives.
    - margin: float, margin for calculating the loss.

    Returns:
    - total_loss: float, the value of the triplet loss.
    """
    total_loss = 0
    
    # Iterate over each anchor and its corresponding positive and negative samples
    for i in range(len(anchors)):
        anchor = anchors[i]
        positive = positives[i]
        
        # Compute the squared distance between the anchor and the positive example
        pos_dist = np.sum(np.square(anchor - positive), axis=-1)
        
        # Compute the Triplet Loss for each negative example
        for negative in negatives:
            neg_dist = np.sum(np.square(anchor - negative), axis=-1)
            total_loss += np.maximum(0, pos_dist - neg_dist + margin)
    
    # Average the loss over the number of anchor-positive pairs
    total_loss /= (len(anchors) * len(negatives))
    
    return total_loss