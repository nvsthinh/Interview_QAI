import numpy as np
import pickle
import config
from utils import *  # Import utility functions (e.g., triplet_loss) from utils module

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the MLP with one hidden layer.

        Parameters:
        - input_size: int, number of input features
        - hidden_size: int, number of neurons in the hidden layer
        - output_size: int, number of output neurons
        """
        # Initialize weights for the hidden layer with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        # Initialize biases for the hidden layer to zero
        self.b1 = np.zeros((1, hidden_size))
        # Initialize weights for the output layer with small random values
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        # Initialize biases for the output layer to zero
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, z):
        """
        Apply the ReLU activation function.

        Parameters:
        - z: numpy array, input to the activation function

        Returns:
        - numpy array with ReLU applied
        """
        return np.maximum(0, z)
    
    def forward(self, X):
        """
        Perform the forward pass of the network.

        Parameters:
        - X: numpy array of shape (n_samples, input_size), input features

        Returns:
        - Output of the network
        """
        # Compute the input to the hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        # Apply ReLU activation function
        self.a1 = self.relu(self.z1)
        # Compute the input to the output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def compute_loss(self, anchor, positive, negative, alpha=0.2):
        """
        Compute the triplet loss using the forward pass outputs.

        Parameters:
        - anchor: numpy array, anchor samples
        - positive: numpy array, positive samples
        - negative: numpy array, negative samples
        - alpha: float, margin for the triplet loss

        Returns:
        - Loss value
        """
        # Forward pass for anchor, positive, and negative samples
        anchor_output = self.forward(anchor)
        positive_output = self.forward(positive)
        negative_output = self.forward(negative)

        # Check for NaN values in outputs to debug potential issues
        if np.isnan(anchor_output).any() or np.isnan(positive_output).any() or np.isnan(negative_output).any():
            print("NaN detected in forward pass outputs")
        
        # Compute the triplet loss using utility function
        loss = triplet_loss(anchor_output, positive_output, negative_output, alpha)
        return loss
    
    def backward(self, anchor, positive, negative, alpha=1.0, learning_rate=0.01):
        """
        Perform the backward pass and update weights using gradient descent.

        Parameters:
        - anchor: numpy array, anchor samples
        - positive: numpy array, positive samples
        - negative: numpy array, negative samples
        - alpha: float, margin for the triplet loss
        - learning_rate: float, learning rate for weight updates
        """
        # Forward pass to get outputs for anchor, positive, and negative samples
        anchor_output = self.forward(anchor)
        positive_output = self.forward(positive)
        negative_output = self.forward(negative)
        
        # Calculate the gradients for the loss with respect to outputs
        pos_dist = 2 * (anchor_output - positive_output)  # Gradient w.r.t. positive distance
        neg_dist = 2 * (anchor_output - negative_output)  # Gradient w.r.t. negative distance
        
        dloss_da = pos_dist - neg_dist  # Gradient w.r.t. anchor output
        dloss_dp = -pos_dist  # Gradient w.r.t. positive output
        dloss_dn = neg_dist  # Gradient w.r.t. negative output

        # Check for NaN values in gradients to debug potential issues
        if np.isnan(dloss_da).any() or np.isnan(dloss_dp).any() or np.isnan(dloss_dn).any():
            print("NaN detected in gradients")

        # Update weights and biases for the output layer
        self.W2 -= learning_rate * np.dot(self.a1.T, dloss_da)  # Gradient descent update for W2
        self.b2 -= learning_rate * np.sum(dloss_da, axis=0, keepdims=True)  # Gradient descent update for b2
        
        # Compute gradients for the hidden layer
        dW1_a = np.dot(anchor.T, np.dot(dloss_da, self.W2.T) * (self.z1 > 0))  # Gradient w.r.t. W1 for anchor
        db1_a = np.sum(np.dot(dloss_da, self.W2.T) * (self.z1 > 0), axis=0, keepdims=True)  # Gradient w.r.t. b1 for anchor
        
        dW1_p = np.dot(positive.T, np.dot(dloss_dp, self.W2.T) * (self.z1 > 0))  # Gradient w.r.t. W1 for positive
        db1_p = np.sum(np.dot(dloss_dp, self.W2.T) * (self.z1 > 0), axis=0, keepdims=True)  # Gradient w.r.t. b1 for positive
        
        dW1_n = np.dot(negative.T, np.dot(dloss_dn, self.W2.T) * (self.z1 > 0))  # Gradient w.r.t. W1 for negative
        db1_n = np.sum(np.dot(dloss_dn, self.W2.T) * (self.z1 > 0), axis=0, keepdims=True)  # Gradient w.r.t. b1 for negative
        
        # Update weights and biases for the hidden layer
        self.W1 -= learning_rate * (dW1_a + dW1_p + dW1_n)  # Gradient descent update for W1
        self.b1 -= learning_rate * (db1_a + db1_p + db1_n)  # Gradient descent update for b1

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            model = MLP(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
            model.W1 = data.W1
            model.b1 = data.b1
            model.W2 = data.W2
            model.b2 = data.b2
            return model