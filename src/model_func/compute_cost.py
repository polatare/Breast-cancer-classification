import numpy as np 
from model_func.sigmoid import sigmoid
def compute_cost(X, y, w, b, lambda_ = 0):
    """
    Computes the cost over all examples
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """
    m,n = X.shape
    cost = 0.
    z_wb = np.dot(X,w) + b
    for i in range(m):
        f_wb = sigmoid(z_wb[i])
        cost += (-y[i]*np.log(f_wb)) - ((1-y[i])*np.log(1-f_wb))
    cost = cost/m 
    reg_cost  = 0.
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost 
    total_cost = cost + reg_cost 
    return total_cost
