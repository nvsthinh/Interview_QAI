# Triplet Loss
## 1. Definition
Triplet Loss is a commonly used loss function in image recognition and matching problems, especially in deep learning models. The goal of Triplet Loss is to ensure that samples of the same class are closer to each other in the feature space than samples of different classes.
## 2. Formula Triplet Loss with One Samples (Question 2 - a)
The Triplet Loss can be expressed as:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \max\left(0, \|f(x_i^a) - f(x_i^p)\|^2 - \|f(x_i^a) - f(x_i^n)\|^2 + \alpha\right)$$

Where
- $x_i^a$ is the anchor.
- $x_i^p$ is the positive sample (same class as the anchor).
- $x_i^n$ is the negative sample (different class as the anchor).
- $α$ is the margin.
- $N$ is the number of triplets used in the loss calculation.


## 3. Formula Triplet Loss with Multiple Samples (Question 2 - b)
$$\mathcal{L} = \frac{1}{A} \sum_{i=1}^{N} \max\left(0, \frac{1}{P} \sum_{p\in P}\|f(x_i^a) - f(x_i^p)\|^2 - \frac{1}{N}\sum_{n\in N} \|f(x_i^a) - f(x_i^n)\|^2 + \alpha\right)$$
Where
- $x_i^a$ is the anchor.
- $x_i^p$ is the positive sample.
- $x_i^n$ is the negative sample.
- $α$ is the margin.
- $P$ is number of positive samples.
- $N$ is number of negative samples.
- $A$ is the number of triplets used in the loss calculation.

## 4. Explain Formula
### 4.1. Embedding Function $f$
- $f(x)$: A function (usually a neural network) that maps an input $x$ to an embedding space where comparisons can be made.
### 4.2. Distance Metric
- $\|f(x_i^a) - f(x_i^p)\|^2$: The squared distance between the anchor and the positive sample in the embedding space. (L2 distance)
- $\|f(x_i^a) - f(x_i^n)\|^2$: The squared distance between the anchor and the negative sample in the embedding space (L2 distance)
### 4.3. Margin $\alpha $
A margin that is enforced between the positive and negative pairs. This margin helps ensure that the negative examples are farther away from the anchor than the positive examples by at least $\alpha$.
### 4.4. Loss Calculation
- If the positive pair distance is not sufficiently smaller than the negative pair distance by at least $\alpha$, the term $\|f(x_i^a) - f(x_i^p)\|^2 - \|f(x_i^a) - f(x_i^n)\|^2 + \alpha$ will be positive, contributing to the loss.
- The $\max$ function ensures that only positive values contribute to the loss. If the difference is negative (i.e., the triplet already satisfies the condition), the loss for that triplet is zero.
## 5. Advantages & Disadvantages
### 5.1. Advantages
- Discriminative Feature Learning
- Robust to Class Imbalance
- Good for High-Dimensional Data

### 5.2. Disadvantages
- Margin Sensitivity
- Selection of Triplets
- Computational Cost
## 6. Applications
- **Face Recognition**: Ensures that faces of the same person are closer in the feature space compared to faces of different people.
- **Image Retrieval**: Helps in learning embeddings such that similar images are closer together in the feature space.

## 7. Files
- `triplet_loss_one_sample.py`: Contain Triplet Loss implementation with 1 anchor and 1 fake sample
- `triplet_loss_multi_samples.py`: Contain Triplet Loss implementation with 2 anchors and 5 fake samples
- `notebook/Triplet_Loss.ipynb`: contain example implementation with one and multiple samples
