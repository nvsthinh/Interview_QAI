from flask import Flask, request, jsonify
import numpy as np
from Question_1.model import AdaBoost
from Question_1.utils import load_data

app = Flask(__name__)

@app.route('/adaboost', methods=['POST'])
def adaboost():
    data = request.get_json()
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    X_train, X_test, y_train, y_test = X_train[:100], X_test[:10], y_train[:100], y_test[:10]
    # Initialize and train model
    model = AdaBoost(n_clf=10)
    model.fit(X_train, y_train)
    
    # Evaluate model
    accuracy = model.evaluate(X_test, y_test)
    
    return jsonify({"accuracy": accuracy})

if __name__ == '__main__':
    app.run(debug=True)