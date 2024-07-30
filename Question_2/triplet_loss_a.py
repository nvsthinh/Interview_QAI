import numpy as np

def triplet_loss_a(anchor, positive, negative, margin=1.0):
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