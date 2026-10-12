import numpy as np
def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column 
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    mu = np.mean(X,axis=0)     # mu will have shape (n,)
    sigma  = np.std(X, axis=0) # sigma will have shape (n,)
    X_norm = (X - mu) / sigma  
    return (X_norm, mu, sigma)