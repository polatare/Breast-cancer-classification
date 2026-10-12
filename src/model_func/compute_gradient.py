import numpy as np 
from model_func.sigmoid import sigmoid
def compute_gradient(X, y, w, b, lambda_):
    """
    Computes the gradient 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    z_wb = np.dot(X,w) + b
    f_wb = sigmoid(z_wb)
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        err = f_wb[i] - y[i]
        dj_db += err
        for j in range(n):
            dj_dw[j] += err * X[i][j]
    dj_dw /= m
    dj_db /= m
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]
    return  dj_dw , dj_db