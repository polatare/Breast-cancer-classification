import numpy as np
from model_func.sigmoid import sigmoid

def predict(X, w, b):
    probabilities = sigmoid(np.dot(X, w) + b)
    predictions = (probabilities >= 0.5).astype(int)
    return predictions
