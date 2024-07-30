# MLP Classifier with MNIST dataset using Numpy library and Triplet Loss
![Question 3 Workflow](https://github.com/nvsthinh/Interview_QAI/blob/main/data/Q3.png)

## 1. MLP Classifier
MLP classification model with Triplet Loss. It has 2 layers with a hidden size of 128 and an output size of 64. Train on 10 epochs to demo API with batch sizes are 32.

## 2. Files
- `model.py`: Implementation MLP Classifier from scratch using Numpy
- `config.py`: Contain configuration such as learning rate, batch size, ...
- `main.py`: Simple pipeline
- `utils.py`: Contain triplet_loss_one_sample, load_data 