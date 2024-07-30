from model import RandomForest
from utils import load_data

def main():
    # Load MNIST data
    X_train, y_train, X_test, y_test = load_data()

    # Initialize the model
    model = RandomForest(n_trees=10, max_depth=10)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()