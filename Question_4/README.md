# MNIST Image Classification API

This repository provides a Flask-based API for classifying MNIST handwritten digits using a Multi-Layer Perceptron (MLP) model implemented with Numpy.

## Features

- Upload an image of a handwritten digit.
- Predict the digit using an MLP model.
- Return the predicted label.

## Prerequisites

Ensure you have the following packages installed:

- Flask
- Numpy
- Pillow

You can install the required packages using pip:

```bash
pip install -m requirements.txt
```
## Project Structure

- `app.py`: Main Flask application file that contains the API routes and logic.
- `checkpoint/model.pkl`: Trained MLP model saved using pickle.
- `README.md`: This documentation file.

## How to Run
Start the Flask application by running:
```bash
python app.py
```