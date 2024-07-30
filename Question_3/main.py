from model import MLP
from utils import load_data
import config
import numpy as np

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model = MLP(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    # Train
    for epoch in range(config.EPOCHS):
        LOSS = []
        for i in range(0, X_train.shape[0], config.BATCH_SIZE):
            # Ensure the batch size is consistent
            end = i + config.BATCH_SIZE
            if end > X_train.shape[0]:
                break
            
            anchor_batch = X_train[i:end]
            positive_batch = X_train[i:end]
            negative_batch = X_train[(i+config.BATCH_SIZE) % X_train.shape[0]: (i+2*config.BATCH_SIZE) % X_train.shape[0]]
            
            if len(negative_batch) < len(anchor_batch):
                continue
            
            loss = model.compute_loss(anchor_batch, positive_batch, negative_batch)
            LOSS.append(loss)
            model.backward(anchor_batch, positive_batch, negative_batch, learning_rate=config.LR)
            
        print(f'Epoch {epoch+1}/{config.EPOCHS}, Loss: {np.mean(LOSS)}')