from model import AdaBoost
from utils import load_data
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    # Sample with 100 training samples, 10 testing samples
    # X_train, X_test, y_train, y_test = X_train[:100], X_test[:10], y_train[:100], y_test[:10]
    model = AdaBoost(n_clf=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
