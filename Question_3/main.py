from model import MLP
from utils import *
import config
import numpy as np

def train():
    # Load the data
    X_train, X_test, y_train, y_test = load_data()

    # Initialize the MLP model
    model = MLP(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    for epoch in range(config.EPOCHS):
        epoch_loss = 0
        num_batches = len(X_train) // config.BATCH_SIZE
        for _ in range(num_batches):
            # Create triplet batches
            anchor, positive, negative = create_triplets(X_train, y_train, config.BATCH_SIZE)
            
            # Compute loss
            loss = model.compute_loss(anchor, positive, negative)
            epoch_loss += np.sum(loss)
            
            # Backward pass and update weights
            model.backward(anchor, positive, negative, config.LR)
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{config.EPOCHS}, Loss: {avg_loss:.4f}")

    print("Training completed.")
    return model

def inference(model, input):
    # Load the data
    X_train, X_test, y_train, y_test = load_data()
    label_features_list, unique_labels = extract_label_features(model, X_train, y_train)

    # Extract features from the input sample
    input_features = model.forward(input)

    # Calculate similarity scores with each label feature
    similarities = [cosine_similarity(input_features, label_feature) for label_feature in label_features_list]
    
    # Find the label with the highest similarity score
    predicted_label = unique_labels[np.argmax(similarities)]
    
    return predicted_label

def main():
    # Example usage
    X_train, X_test, y_train, y_test = load_data()
    model = train()
    input_sample = X_test[0]  # Example input sample from test set
    predicted_label = inference(model, input_sample)
    print(f"Predicted Label: {predicted_label}, True Label: {y_test[0]}")

if __name__ == "__main__":
    main()