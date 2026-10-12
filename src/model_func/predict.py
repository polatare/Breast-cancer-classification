import numpy as np
from model_func.sigmoid import sigmoid
def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w , b
    
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    m = X.shape[0]  
    p = np.zeros(m)
    z_wb = np.dot(X,w) + b
    f_wb = sigmoid(z_wb)
    for i in range(m):   
        p[i] = 1 if f_wb[i] >= 0.4 else 0
    return p