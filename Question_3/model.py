import numpy as np
from utils import *

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def compute_loss(self, anchor, positive, negative, alpha=0.2):
        anchor_output = self.forward(anchor)
        positive_output = self.forward(positive)
        negative_output = self.forward(negative)

        # Ensure there are no NaN values in outputs
        if np.isnan(anchor_output).any() or np.isnan(positive_output).any() or np.isnan(negative_output).any():
            print("NaN detected in forward pass outputs")
        
        loss = triplet_loss(anchor_output, positive_output, negative_output, alpha)
        return loss
    
    def backward(self, anchor, positive, negative, alpha=0.2, learning_rate=0.01):
        # Forward pass
        anchor_output = self.forward(anchor)
        positive_output = self.forward(positive)
        negative_output = self.forward(negative)
        
        # Calculate gradients (simple backpropagation)
        pos_dist = 2 * (anchor_output - positive_output)
        neg_dist = 2 * (anchor_output - negative_output)
        
        dloss_da = pos_dist - neg_dist
        dloss_dp = -pos_dist
        dloss_dn = neg_dist

        # Ensure there are no NaN values in gradients
        if np.isnan(dloss_da).any() or np.isnan(dloss_dp).any() or np.isnan(dloss_dn).any():
            print("NaN detected in gradients")

        # Update weights and biases (simplified gradient descent)
        self.W2 -= learning_rate * np.dot(self.a1.T, dloss_da)
        self.b2 -= learning_rate * np.sum(dloss_da, axis=0, keepdims=True)
        
        dW1_a = np.dot(anchor.T, np.dot(dloss_da, self.W2.T) * (self.z1 > 0))
        db1_a = np.sum(np.dot(dloss_da, self.W2.T) * (self.z1 > 0), axis=0, keepdims=True)
        
        dW1_p = np.dot(positive.T, np.dot(dloss_dp, self.W2.T) * (self.z1 > 0))
        db1_p = np.sum(np.dot(dloss_dp, self.W2.T) * (self.z1 > 0), axis=0, keepdims=True)
        
        dW1_n = np.dot(negative.T, np.dot(dloss_dn, self.W2.T) * (self.z1 > 0))
        db1_n = np.sum(np.dot(dloss_dn, self.W2.T) * (self.z1 > 0), axis=0, keepdims=True)
        
        self.W1 -= learning_rate * (dW1_a + dW1_p + dW1_n)
        self.b1 -= learning_rate * (db1_a + db1_p + db1_n)