import numpy as np
def Data_Split(X,y):
    # 1. Get the indices for each class
    benign_indices = np.where(y == 0)[0]
    # say 357 
    malignant_indices = np.where(y == 1)[0]
    # say 150

    # 2. Shuffle each class independently
    np.random.seed(42)
    np.random.shuffle(benign_indices)
    np.random.shuffle(malignant_indices)

    # 3. Split each class: 80% train, 20% test
    benign_split = int(0.8 * len(benign_indices))
    malignant_split = int(0.8 * len(malignant_indices))

    benign_train = benign_indices[:benign_split]
    benign_test = benign_indices[benign_split:]

    malignant_train = malignant_indices[:malignant_split]
    malignant_test = malignant_indices[malignant_split:]

    # 4. Combine the indices
    train_indices = np.concatenate([benign_train, malignant_train])
    test_indices = np.concatenate([benign_test, malignant_test])

    # 5. Shuffle the combined indices
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # 6. Select the matching rows from X and y
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    return (X_train,X_test,y_train,y_test)