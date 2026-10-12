# Breast Cancer Classification

A binary classifier (benign vs malignant) built from scratch with NumPy, plus a PyQt5 desktop app to run and inspect it. The model is logistic regression trained with batch gradient descent, with no machine learning framework doing the learning.

> Educational project. It is not a medical device and must not be used for diagnosis.

## Data and features

The data is the Wisconsin breast cancer dataset (`data/breast_cancer_edited.csv`, no header row, label in the second column: `0` = benign, `1` = malignant). Five of the 30 measurements are used:

| Feature | Meaning |
|---|---|
| `concave_points_worst` | Largest number of concave portions of the nucleus contour in the image |
| `radius_worst` | Largest mean distance from the nucleus center to its perimeter |
| `concavity_worst` | Largest severity of the concave portions of the contour |
| `area_worst` | Largest nucleus area in the image |
| `texture_mean` | Mean standard deviation of gray-scale values inside the nucleus |

## How the model works

1. **Split** (`Data_handeling/split_data.py`): stratified 80/20 train/test split per class, seed 42.
2. **Scaling** (`Data_handeling/z_score.py`): z-score normalization. Mean and standard deviation come from the training set only and are re-used on the test set and on GUI inputs.
3. **Training** (`model_func/`): sigmoid, cross-entropy cost, gradient, and batch gradient descent with L2 regularization to control overfitting.
4. **Decision threshold**: a sample is malignant when its probability is at or above the threshold. Lowering it raises recall (fewer missed malignant cases) at the cost of precision.

Default settings: learning rate `0.01`, `1000` iterations, lambda `1`, GUI threshold `0.40`.

## Project structure

```
breast-cancer-classification/
├── data/                 breast_cancer.csv, breast_cancer_edited.csv
├── notebooks/            01_data_exploration.ipynb
├── src/
│   ├── gui.py            PyQt5 application (entry point for the GUI)
│   ├── main.py           command line training and threshold report
│   ├── accurcy.py        prediction helper at a fixed 0.5 threshold
│   ├── Data_handeling/   split_data.py, z_score.py
│   └── model_func/       sigmoid, compute_cost, compute_gradient,
│                         gradient_descent, predict
├── tests/                empty for now
├── documentation.md
└── ARCHITECTURE.txt
```

## Running

Requirements: Python 3.9+ and

```
pip install numpy pandas PyQt5 scikit-learn
```

(`scikit-learn` is only needed by `main.py`; the GUI computes its own metrics.)

```
python src/gui.py      # desktop app
python src/main.py     # console report for thresholds 0.5, 0.4, 0.3, 0.2
```

`main.py` reads the dataset from a hard-coded Windows path; edit it if your data lives elsewhere. The GUI finds the data through a path relative to the project and offers a file picker if it is missing.

## The GUI

The app opens full screen without a window frame (kiosk mode). The model trains in the background at start-up, which takes a few seconds. Use **Exit application** in the sidebar or `Ctrl+Q` to close it.

| Page | Purpose |
|---|---|
| Prediction | Enter the five features, get the class and malignancy probability. Can load a random benign or malignant test sample. |
| Performance | Accuracy, precision, recall, specificity, F1 and the confusion matrix for the chosen threshold, plus a comparison table across thresholds. |
| Model Details | Training configuration, split sizes, learned weights and bias, and the cost curve. |
| Features | Plain-language meaning and training-set range of each feature. |

The decision threshold slider in the sidebar is shared by every page. Hovering over a feature name or input field shows what that feature means.

## Possible next steps

- Add unit tests in `tests/` for the split, scaling and gradient functions.
- Vectorize `compute_gradient` and `compute_cost` to speed up training.
- Add ROC and precision-recall curves to the Performance page.
