import numpy as np 
import copy
from model_func.compute_gradient import compute_gradient
from model_func.compute_cost import compute_cost
def gradient_descent(X, y, w_in, b_in, alpha, num_iters, lambda_):
    """
    Performs batch gradient descent
    
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial value/s of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent
      lambda             : overfitting handeler
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    # Initialize variables
    w = copy.deepcopy(w_in) 
    b = b_in
    J_history = []
    for i in range(num_iters):
        dj_dw , dj_db = compute_gradient(X,y,w ,b , lambda_)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        # Save cost J at each iteration
        if i < 10000:
            J_history.append(compute_cost(X ,y ,w ,b,lambda_))
    return w, b , J_history