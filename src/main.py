import numpy as np
import pandas as pd
from Data_handeling import z_score as zs
from Data_handeling import split_data as sd
from model_func.predict import predict
from model_func.compute_gradient import compute_gradient
from model_func.gradient_descent import gradient_descent
from model_func.sigmoid import sigmoid
import accurcy as acc
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import classification_report
columns = [
    "id", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean",
    "area_mean", "smoothness_mean", "compactness_mean",
    "concavity_mean", "concave_points_mean", "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave_points_worst",
    "symmetry_worst", "fractal_dimension_worst"
]
df = pd.read_csv(
    "D:\\Python\\breast-cancer\\data\\breast_cancer_edited.csv",
    header=None,
    names=columns
)
y_train = np.array(df["diagnosis"])
X_train = np.array(df[["concave_points_worst", "radius_worst" , "concavity_worst" , "area_worst" , "texture_mean"]])
X_train, X_test, y_train, y_test = sd.Data_Split(X_train,y_train)
X_norm_train , mu , sigma = zs.zscore_normalize_features(X_train)
X_norm_test = (X_test - mu) / sigma

m, n = X_norm_train.shape

w = np.zeros(n)
b = 0.0

P = predict(X_norm_train , w , b)


w , b , J = gradient_descent(X_norm_train,y_train,w,b,0.01,1000,1)


y_pred = acc.predict(X_norm_test, w, b)

# print("Predictions:", y_pred)
# print("Actual:", y_test)


# accuracy = np.mean(y_pred == y_test) * 100

# print(f"Test Accuracy: {accuracy:.2f}%")


cm = confusion_matrix(y_test, y_pred)

# print(cm)
# print(classification_report(y_test, y_pred))
probabilities = sigmoid(np.dot(X_norm_test, w) + b)

for threshold in [0.5, 0.4, 0.3, 0.2]:
    y_pred = (probabilities >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))