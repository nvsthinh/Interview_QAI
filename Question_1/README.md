# 1. Adaboost Classifier with MNIST dataset
AdaBoost, short for Adaptive Boosting, is an ensemble learning method that combines the predictions of multiple weak classifiers to produce a strong classifier. It is primarily used for classification tasks. The core idea of AdaBoost is to iteratively train weak classifiers on the data, each time adjusting the weights of the training samples to focus more on those that were misclassified by previous classifiers. The final prediction is made by a weighted majority vote of the weak classifiers.

- See more detail explain Adaboost model in: [Bài 6 - Adaboost [Thịnh Diablog]](https://flowery-fairy-f0d.notion.site/B-i-6-AdaBoost-14583006b8084791967b24c74a29a2b3?pvs=4)

# 2. Files
- `main.py`: Simple pipeline using AdaBoost Classifier in MNIST dataset
- `model.py`: Adaboost implementation from scratch using Numpy
- `utils.py`: Load MNIST dataset
- `notebook/AdaBoost_with_MNIST_Dataset.ipynb`: Example of implementation for Question 1