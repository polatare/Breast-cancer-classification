"""
Breast Cancer Classification - desktop GUI (PyQt5)

Run from the project root or from src/:
    python src/gui.py

The GUI re-uses the project's own modules (Data_Split, zscore_normalize_features,
gradient_descent, sigmoid), trains the model on start-up in a background thread,
and then exposes four pages:

    1. Prediction     - enter the 5 features and get a result
    2. Performance    - accuracy, confusion matrix and metrics for the chosen threshold
    3. Model Details  - training setup, learned parameters, cost curve
    4. Features       - what each feature means, with training-set statistics

The decision threshold is shared by all pages and is controlled from the sidebar.
Labels: 0 = benign, 1 = malignant.
"""
import os
import sys

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QRectF, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QKeySequence, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QDoubleSpinBox, QFileDialog,
    QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QProgressBar,
    QPushButton, QScrollArea, QShortcut, QSlider, QStackedWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from Data_handeling.split_data import Data_Split              # noqa: E402
from Data_handeling.z_score import zscore_normalize_features  # noqa: E402
from model_func.gradient_descent import gradient_descent      # noqa: E402
from model_func.sigmoid import sigmoid                        # noqa: E402

# --------------------------------------------------------------------------- #
# Project configuration (mirrors main.py)
# --------------------------------------------------------------------------- #
DATA_PATH = os.path.join(ROOT_DIR, "data", "breast_cancer_edited.csv")

COLUMNS = [
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
    "symmetry_worst", "fractal_dimension_worst",
]

FEATURES = [
    "concave_points_worst", "radius_worst", "concavity_worst",
    "area_worst", "texture_mean",
]

ALPHA = 0.01        # learning rate
ITERATIONS = 1000   # gradient descent iterations
LAMBDA = 1          # L2 regularization strength
DEFAULT_THRESHOLD = 0.40
SWEEP_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

FEATURE_INFO = {
    "concave_points_worst": {
        "title": "Concave Points (worst)",
        "meaning": ("The largest value, across the image, of the number of concave "
                    "portions of the nucleus contour. More concave points means a "
                    "more irregular nucleus border, which is typical of malignant cells."),
    },
    "radius_worst": {
        "title": "Radius (worst)",
        "meaning": ("The largest mean distance from the center of a nucleus to the points "
                    "on its perimeter. A bigger worst radius means larger nuclei, which "
                    "is associated with malignancy."),
    },
    "concavity_worst": {
        "title": "Concavity (worst)",
        "meaning": ("The largest severity of the concave portions of the nucleus contour. "
                    "It measures how deep the inward dents are, not just how many there are."),
    },
    "area_worst": {
        "title": "Area (worst)",
        "meaning": ("The largest nucleus area found in the image. It grows together with "
                    "the radius and is one of the strongest separators between benign "
                    "and malignant samples."),
    },
    "texture_mean": {
        "title": "Texture (mean)",
        "meaning": ("The mean standard deviation of gray-scale values inside the nucleus. "
                    "Higher values mean a more uneven, grainy internal appearance."),
    },
}


def feature_tooltip(name):
    info = FEATURE_INFO[name]
    return ('<table width="330"><tr><td><b>%s</b><br>%s</td></tr></table>'
            % (info["title"], info["meaning"]))


# --------------------------------------------------------------------------- #
# Model training and metrics (pure numpy, no GUI code)
# --------------------------------------------------------------------------- #
def load_dataset(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    diagnosis = df["diagnosis"]
    if diagnosis.dtype == object:  # tolerate M / B letters
        y = diagnosis.astype(str).str.strip().str.upper().map({"M": 1, "B": 0})
    else:
        y = diagnosis
    y = y.to_numpy().astype(int)
    X = df[FEATURES].to_numpy(dtype=float)
    return X, y


def train_model(path):
    X, y = load_dataset(path)
    X_train, X_test, y_train, y_test = Data_Split(X, y)
    X_norm_train, mu, sigma = zscore_normalize_features(X_train)
    X_norm_test = (X_test - mu) / sigma

    w0 = np.zeros(X_norm_train.shape[1])
    w, b, J = gradient_descent(X_norm_train, y_train, w0, 0.0, ALPHA, ITERATIONS, LAMBDA)

    return {
        "w": w, "b": float(b), "mu": mu, "sigma": sigma, "J": list(J),
        "X_test": X_test, "y_test": y_test, "y_train": y_train,
        "probs_test": sigmoid(np.dot(X_norm_test, w) + b),
        "n_train": len(y_train), "n_test": len(y_test),
        "data_min": X.min(axis=0), "data_max": X.max(axis=0),
        "train_min": X_train.min(axis=0), "train_max": X_train.max(axis=0),
    }


def compute_metrics(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    tn = int(np.sum((y_true == 0) & (pred == 0)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    total = max(tn + fp + fn + tp, 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tn + tp) / total, "precision": precision,
        "recall": recall, "specificity": specificity, "f1": f1,
    }


class TrainWorker(QThread):
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            self.done.emit(train_model(self.path))
        except Exception as exc:  # shown to the user on the loading page
            self.failed.emit("%s: %s" % (type(exc).__name__, exc))


# --------------------------------------------------------------------------- #
# Small reusable widgets
# --------------------------------------------------------------------------- #
def make_card():
    frame = QFrame()
    frame.setObjectName("card")
    return frame


def page_header(title, subtitle):
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    t = QLabel(title)
    t.setObjectName("pageTitle")
    s = QLabel(subtitle)
    s.setObjectName("pageSubtitle")
    s.setWordWrap(True)
    lay.addWidget(t)
    lay.addWidget(s)
    return box


class MetricCard(QFrame):
    def __init__(self, title, tooltip=""):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(2)
        self.title = QLabel(title)
        self.title.setObjectName("metricTitle")
        self.value = QLabel("-")
        self.value.setObjectName("metricValue")
        lay.addWidget(self.title)
        lay.addWidget(self.value)
        if tooltip:
            self.setToolTip(tooltip)

    def set_value(self, text):
        self.value.setText(text)


class ConfusionMatrixWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.counts = (0, 0, 0, 0)
        self.setMinimumSize(460, 380)

    def set_counts(self, tn, fp, fn, tp):
        self.counts = (tn, fp, fn, tp)
        self.update()

    def paintEvent(self, event):
        tn, fp, fn, tp = self.counts
        grid = [[tn, fp], [fn, tp]]
        names = [["True negative", "False positive"], ["False negative", "True positive"]]
        peak = max(1, tn, fp, fn, tp)

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        left, top, right, bottom = 150, 80, 10, 10
        cw = (self.width() - left - right) / 2.0
        ch = (self.height() - top - bottom) / 2.0

        title_font = QFont(self.font())
        title_font.setPointSize(13)
        title_font.setBold(True)
        p.setFont(title_font)
        p.setPen(QColor("#111827"))
        p.drawText(QRectF(0, 0, self.width(), 30), int(Qt.AlignCenter), "Confusion matrix")

        head_font = QFont(self.font())
        head_font.setPointSize(11)
        head_font.setBold(True)
        p.setFont(head_font)
        p.setPen(QColor("#374151"))
        p.drawText(QRectF(left, 44, cw, 30), int(Qt.AlignCenter), "Predicted: Benign")
        p.drawText(QRectF(left + cw, 44, cw, 30), int(Qt.AlignCenter), "Predicted: Malignant")
        flags = int(Qt.AlignVCenter | Qt.AlignRight | Qt.TextWordWrap)
        p.drawText(QRectF(8, top, left - 20, ch), flags, "Actual: Benign")
        p.drawText(QRectF(8, top + ch, left - 20, ch), flags, "Actual: Malignant")

        count_font = QFont(self.font())
        count_font.setPointSize(26)
        count_font.setBold(True)
        small_font = QFont(self.font())
        small_font.setPointSize(11)

        for r in range(2):
            for c in range(2):
                color = QColor("#16a34a" if r == c else "#dc2626")
                color.setAlpha(int(45 + 190 * grid[r][c] / peak))
                rect = QRectF(left + c * cw + 4, top + r * ch + 4, cw - 8, ch - 8)
                p.setPen(Qt.NoPen)
                p.setBrush(color)
                p.drawRoundedRect(rect, 10, 10)

                p.setPen(QColor("#111827"))
                p.setFont(count_font)
                p.drawText(QRectF(rect.left(), rect.top() + rect.height() * 0.18,
                                  rect.width(), rect.height() * 0.45),
                           int(Qt.AlignCenter), str(grid[r][c]))
                p.setFont(small_font)
                p.drawText(QRectF(rect.left(), rect.top() + rect.height() * 0.62,
                                  rect.width(), rect.height() * 0.25),
                           int(Qt.AlignCenter), names[r][c])
        p.end()


class CostCurveWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.values = []
        self.setMinimumSize(520, 340)

    def set_values(self, values):
        self.values = list(values)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("white"))
        p.setPen(QColor("#111827"))
        title_font = QFont(self.font())
        title_font.setBold(True)
        title_font.setPointSize(12)
        p.setFont(title_font)
        p.drawText(QRectF(0, 6, self.width(), 26), int(Qt.AlignCenter),
                   "Training cost per iteration")
        p.setFont(self.font())

        n = len(self.values)
        plot = QRectF(80, 46, self.width() - 110, self.height() - 100)
        p.setPen(QPen(QColor("#9ca3af"), 1))
        p.drawLine(int(plot.left()), int(plot.bottom()), int(plot.right()), int(plot.bottom()))
        p.drawLine(int(plot.left()), int(plot.top()), int(plot.left()), int(plot.bottom()))
        if n < 2:
            p.end()
            return

        lo, hi = min(self.values), max(self.values)
        span = (hi - lo) or 1.0
        path = QPainterPath()
        step = max(1, n // 400)
        indices = list(range(0, n, step))
        if indices[-1] != n - 1:
            indices.append(n - 1)
        for k, i in enumerate(indices):
            x = plot.left() + plot.width() * i / (n - 1)
            y = plot.bottom() - plot.height() * (self.values[i] - lo) / span
            if k == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        p.setPen(QPen(QColor("#2563eb"), 2))
        p.drawPath(path)

        p.setPen(QColor("#374151"))
        p.drawText(QRectF(4, plot.top() - 10, 72, 20), int(Qt.AlignRight | Qt.AlignVCenter),
                   "%.4f" % hi)
        p.drawText(QRectF(4, plot.bottom() - 10, 72, 20), int(Qt.AlignRight | Qt.AlignVCenter),
                   "%.4f" % lo)
        p.drawText(QRectF(plot.left() - 20, plot.bottom() + 6, 80, 20),
                   int(Qt.AlignLeft | Qt.AlignVCenter), "iteration 0")
        p.drawText(QRectF(plot.right() - 100, plot.bottom() + 6, 120, 20),
                   int(Qt.AlignRight | Qt.AlignVCenter), "iteration %d" % n)
        p.end()


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
class PredictionPage(QWidget):
    def __init__(self, model, threshold):
        super().__init__()
        self.model = model
        self.threshold = threshold
        self.last_probability = None
        self.spins = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 28)
        root.setSpacing(18)
        root.addWidget(page_header(
            "Prediction",
            "Enter the five measurements below and run the model. Hover over any "
            "feature name or field to see what it means."))

        body = QHBoxLayout()
        body.setSpacing(20)
        root.addLayout(body, 1)

        # ---- form card
        form = make_card()
        fl = QVBoxLayout(form)
        fl.setContentsMargins(24, 20, 24, 20)
        fl.setSpacing(14)
        heading = QLabel("Feature entry")
        heading.setObjectName("cardTitle")
        fl.addWidget(heading)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(14)
        for row, name in enumerate(FEATURES):
            tip = feature_tooltip(name)
            label = QLabel(FEATURE_INFO[name]["title"])
            label.setToolTip(tip)
            label.setObjectName("fieldLabel")

            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(0.0, float(model["data_max"][row]) * 2.0)
            spin.setSingleStep(max(round(float(model["data_max"][row]) * 0.01, 4), 0.001))
            spin.setToolTip(tip)
            spin.setMinimumWidth(170)
            spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
            spin.valueChanged.connect(self._clear_status)
            self.spins[name] = spin

            rng = QLabel("training range: %.3f to %.3f" % (
                model["train_min"][row], model["train_max"][row]))
            rng.setObjectName("hint")
            rng.setToolTip(tip)

            grid.addWidget(label, row, 0)
            grid.addWidget(spin, row, 1)
            grid.addWidget(rng, row, 2)
        fl.addLayout(grid)

        buttons = QHBoxLayout()
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.setObjectName("primary")
        self.predict_btn.clicked.connect(self.run_prediction)
        benign_btn = QPushButton("Load benign test sample")
        benign_btn.clicked.connect(lambda: self.load_sample(0))
        malignant_btn = QPushButton("Load malignant test sample")
        malignant_btn.clicked.connect(lambda: self.load_sample(1))
        reset_btn = QPushButton("Reset to averages")
        reset_btn.clicked.connect(self.reset_values)
        for b in (self.predict_btn, benign_btn, malignant_btn, reset_btn):
            buttons.addWidget(b)
        fl.addLayout(buttons)

        self.status = QLabel("")
        self.status.setObjectName("hint")
        fl.addWidget(self.status)
        fl.addStretch(1)
        body.addWidget(form, 3)

        # ---- result card
        result = make_card()
        rl = QVBoxLayout(result)
        rl.setContentsMargins(24, 20, 24, 20)
        rl.setSpacing(12)
        rh = QLabel("Result")
        rh.setObjectName("cardTitle")
        rl.addWidget(rh)

        self.result_label = QLabel("No prediction yet")
        self.result_label.setObjectName("resultText")
        self.result_label.setAlignment(Qt.AlignCenter)
        rl.addWidget(self.result_label)

        self.prob_label = QLabel("")
        self.prob_label.setAlignment(Qt.AlignCenter)
        self.prob_label.setObjectName("fieldLabel")
        rl.addWidget(self.prob_label)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(16)
        rl.addWidget(self.bar)

        self.thr_label = QLabel("")
        self.thr_label.setObjectName("hint")
        self.thr_label.setAlignment(Qt.AlignCenter)
        self.thr_label.setWordWrap(True)
        rl.addWidget(self.thr_label)

        rl.addStretch(1)
        note = QLabel("This tool is a course project for learning purposes. "
                      "It is not a medical device and must not be used for diagnosis.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        rl.addWidget(note)
        body.addWidget(result, 2)

        self.reset_values()
        self._render()

    # ---- helpers
    def _set_values(self, values):
        for spin, v in zip(self.spins.values(), values):
            spin.blockSignals(True)
            spin.setValue(float(v))
            spin.blockSignals(False)

    def _clear_status(self):
        self.status.setText("")

    def reset_values(self):
        self._set_values(self.model["mu"])
        self.status.setText("Fields set to the training-set averages.")

    def load_sample(self, cls):
        idx = np.where(self.model["y_test"] == cls)[0]
        if len(idx) == 0:
            return
        i = int(np.random.choice(idx))
        self._set_values(self.model["X_test"][i])
        self.status.setText("Loaded a random %s sample from the held-out test set."
                            % ("malignant" if cls == 1 else "benign"))

    def run_prediction(self):
        x = np.array([self.spins[n].value() for n in FEATURES], dtype=float)
        z = (x - self.model["mu"]) / self.model["sigma"]
        self.last_probability = float(sigmoid(np.dot(z, self.model["w"]) + self.model["b"]))
        self._render()

    def set_threshold(self, threshold):
        self.threshold = threshold
        self._render()

    def _render(self):
        p = self.last_probability
        if p is None:
            self.result_label.setText("No prediction yet")
            self.result_label.setStyleSheet("color: #6b7280;")
            self.prob_label.setText("")
            self.bar.setValue(0)
            self.thr_label.setText("Decision threshold: %.2f" % self.threshold)
            return
        malignant = p >= self.threshold
        self.result_label.setText("MALIGNANT" if malignant else "BENIGN")
        self.result_label.setStyleSheet(
            "color: %s;" % ("#b91c1c" if malignant else "#15803d"))
        self.prob_label.setText("Probability of malignancy: %.1f%%" % (p * 100))
        self.bar.setValue(int(p * 1000))
        self.bar.setStyleSheet(
            "QProgressBar::chunk { background: %s; border-radius: 6px; }"
            % ("#dc2626" if malignant else "#16a34a"))
        self.thr_label.setText(
            "Decision threshold: %.2f. A sample is classified as malignant when its "
            "probability is at or above this value." % self.threshold)


class PerformancePage(QWidget):
    def __init__(self, model, threshold):
        super().__init__()
        self.model = model

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 28)
        root.setSpacing(18)
        root.addWidget(page_header(
            "Performance",
            "Results on the held-out test set. Change the decision threshold in the "
            "sidebar to see how the confusion matrix and metrics respond."))

        cards = QHBoxLayout()
        cards.setSpacing(14)
        self.cards = {
            "accuracy": MetricCard("Accuracy", "Share of all test samples classified correctly."),
            "precision": MetricCard("Precision", "Of the samples predicted malignant, the share that really are."),
            "recall": MetricCard("Recall (sensitivity)", "Of the truly malignant samples, the share the model catches."),
            "specificity": MetricCard("Specificity", "Of the truly benign samples, the share the model clears."),
            "f1": MetricCard("F1 score", "Harmonic mean of precision and recall."),
        }
        for c in self.cards.values():
            cards.addWidget(c)
        root.addLayout(cards)

        body = QHBoxLayout()
        body.setSpacing(20)
        root.addLayout(body, 1)

        left = make_card()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(16, 16, 16, 16)
        self.matrix = ConfusionMatrixWidget()
        ll.addWidget(self.matrix, 1)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setObjectName("fieldLabel")
        ll.addWidget(self.summary)
        body.addWidget(left, 3)

        right = make_card()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(20, 16, 20, 16)
        rh = QLabel("Threshold comparison")
        rh.setObjectName("cardTitle")
        rl.addWidget(rh)
        self.table = QTableWidget(len(SWEEP_THRESHOLDS), 5)
        self.table.setHorizontalHeaderLabels(
            ["Threshold", "Accuracy", "Precision", "Recall", "F1"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for r, t in enumerate(SWEEP_THRESHOLDS):
            m = compute_metrics(model["y_test"], model["probs_test"], t)
            values = ["%.2f" % t, "%.2f%%" % (m["accuracy"] * 100),
                      "%.3f" % m["precision"], "%.3f" % m["recall"], "%.3f" % m["f1"]]
            for c, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)
        rl.addWidget(self.table, 1)
        hint = QLabel("Lowering the threshold catches more malignant cases (higher recall) "
                      "at the cost of more false alarms (lower precision).")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        rl.addWidget(hint)
        body.addWidget(right, 2)

        self.set_threshold(threshold)

    def set_threshold(self, threshold):
        m = compute_metrics(self.model["y_test"], self.model["probs_test"], threshold)
        self.cards["accuracy"].set_value("%.2f%%" % (m["accuracy"] * 100))
        self.cards["precision"].set_value("%.3f" % m["precision"])
        self.cards["recall"].set_value("%.3f" % m["recall"])
        self.cards["specificity"].set_value("%.3f" % m["specificity"])
        self.cards["f1"].set_value("%.3f" % m["f1"])
        self.matrix.set_counts(m["tn"], m["fp"], m["fn"], m["tp"])
        self.summary.setText(
            "Threshold %.2f on %d test samples: %d malignant cases missed (false negatives), "
            "%d benign cases flagged as malignant (false positives)."
            % (threshold, self.model["n_test"], m["fn"], m["fp"]))
        self.table.clearSelection()
        for r, t in enumerate(SWEEP_THRESHOLDS):
            if abs(t - threshold) < 1e-9:
                self.table.selectRow(r)


class DetailsPage(QWidget):
    def __init__(self, model):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 28)
        root.setSpacing(18)
        root.addWidget(page_header(
            "Model Details",
            "How the model was trained and what it learned. Weights are expressed in "
            "z-score normalized feature space."))

        body = QHBoxLayout()
        body.setSpacing(20)
        root.addLayout(body, 1)

        left = QVBoxLayout()
        left.setSpacing(18)
        body.addLayout(left, 2)

        cfg = make_card()
        cl = QGridLayout(cfg)
        cl.setContentsMargins(24, 18, 24, 18)
        cl.setHorizontalSpacing(24)
        cl.setVerticalSpacing(8)
        heading = QLabel("Training configuration")
        heading.setObjectName("cardTitle")
        cl.addWidget(heading, 0, 0, 1, 2)
        y_train = model["y_train"]
        J = model["J"]
        rows = [
            ("Algorithm", "Logistic regression, batch gradient descent"),
            ("Learning rate", str(ALPHA)),
            ("Iterations", str(ITERATIONS)),
            ("Regularization (L2) lambda", str(LAMBDA)),
            ("Feature scaling", "Z-score (train statistics only)"),
            ("Split", "80% train / 20% test, stratified by class, seed 42"),
            ("Training samples", "%d (%d benign, %d malignant)" % (
                model["n_train"], int(np.sum(y_train == 0)), int(np.sum(y_train == 1)))),
            ("Test samples", "%d" % model["n_test"]),
            ("Cost: first / last iteration", "%.4f / %.4f" % (J[0], J[-1])),
        ]
        for r, (k, v) in enumerate(rows, start=1):
            kl = QLabel(k)
            kl.setObjectName("hint")
            vl = QLabel(v)
            vl.setObjectName("fieldLabel")
            cl.addWidget(kl, r, 0)
            cl.addWidget(vl, r, 1)
        left.addWidget(cfg)

        params = make_card()
        pl = QVBoxLayout(params)
        pl.setContentsMargins(20, 16, 20, 16)
        ph = QLabel("Learned parameters")
        ph.setObjectName("cardTitle")
        pl.addWidget(ph)
        table = QTableWidget(len(FEATURES) + 1, 4)
        table.setHorizontalHeaderLabels(["Feature", "Train mean", "Train std", "Weight"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for r, name in enumerate(FEATURES):
            vals = [FEATURE_INFO[name]["title"], "%.4f" % model["mu"][r],
                    "%.4f" % model["sigma"][r], "%+.4f" % model["w"][r]]
            for c, text in enumerate(vals):
                item = QTableWidgetItem(text)
                item.setToolTip(feature_tooltip(name))
                if c:
                    item.setTextAlignment(Qt.AlignCenter)
                table.setItem(r, c, item)
        bias_vals = ["Bias (b)", "-", "-", "%+.4f" % model["b"]]
        for c, text in enumerate(bias_vals):
            item = QTableWidgetItem(text)
            if c:
                item.setTextAlignment(Qt.AlignCenter)
            table.setItem(len(FEATURES), c, item)
        pl.addWidget(table)
        left.addWidget(params, 1)

        right = make_card()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(16, 16, 16, 16)
        curve = CostCurveWidget()
        curve.set_values(J)
        rl.addWidget(curve, 1)
        body.addWidget(right, 3)


class FeaturesPage(QWidget):
    def __init__(self, model):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 28, 32, 28)
        outer.setSpacing(18)
        outer.addWidget(page_header(
            "Features",
            "The five measurements the model uses. They come from digitized images of "
            "a fine needle aspirate of a breast mass; 'worst' means the largest value "
            "found in the image."))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        holder = QWidget()
        hl = QVBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(12)

        for i, name in enumerate(FEATURES):
            info = FEATURE_INFO[name]
            card = make_card()
            card.setToolTip(feature_tooltip(name))
            cl = QVBoxLayout(card)
            cl.setContentsMargins(22, 14, 22, 14)
            cl.setSpacing(4)
            title = QLabel("%s    (%s)" % (info["title"], name))
            title.setObjectName("cardTitle")
            meaning = QLabel(info["meaning"])
            meaning.setWordWrap(True)
            meaning.setObjectName("fieldLabel")
            stats = QLabel("Training data: min %.4f   mean %.4f   max %.4f   |   learned weight %+.4f"
                           % (model["train_min"][i], model["mu"][i],
                              model["train_max"][i], model["w"][i]))
            stats.setObjectName("hint")
            cl.addWidget(title)
            cl.addWidget(meaning)
            cl.addWidget(stats)
            hl.addWidget(card)
        hl.addStretch(1)
        scroll.setWidget(holder)
        outer.addWidget(scroll, 1)


class LoadingPage(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(16)
        self.title = QLabel("Breast Cancer Classification")
        self.title.setObjectName("loadTitle")
        self.title.setAlignment(Qt.AlignCenter)
        self.status = QLabel("Training the model, please wait...")
        self.status.setObjectName("pageSubtitle")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setWordWrap(True)
        self.status.setMaximumWidth(700)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setTextVisible(False)
        self.bar.setFixedSize(420, 14)
        self.choose_btn = QPushButton("Choose data file")
        self.exit_btn = QPushButton("Exit")
        self.choose_btn.hide()
        self.exit_btn.hide()
        lay.addWidget(self.title, 0, Qt.AlignCenter)
        lay.addWidget(self.status, 0, Qt.AlignCenter)
        lay.addWidget(self.bar, 0, Qt.AlignCenter)
        lay.addWidget(self.choose_btn, 0, Qt.AlignCenter)
        lay.addWidget(self.exit_btn, 0, Qt.AlignCenter)

    def show_busy(self):
        self.status.setText("Training the model, please wait...")
        self.bar.show()
        self.choose_btn.hide()
        self.exit_btn.hide()

    def show_error(self, message):
        self.status.setText("Could not train the model.\n%s" % message)
        self.bar.hide()
        self.choose_btn.show()
        self.exit_btn.show()


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
STYLE = """
QWidget { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 14px; color: #111827; }
#appRoot { background: #f3f4f6; }
#sidebar { background: #111827; }
#sidebar QLabel { color: #e5e7eb; }
#brand { font-size: 18px; font-weight: 700; color: white; }
#brandSub { font-size: 12px; color: #9ca3af; }
#navButton { background: transparent; color: #d1d5db; text-align: left; padding: 12px 16px;
             border: none; border-radius: 8px; font-size: 15px; }
#navButton:hover { background: #1f2937; }
#navButton:checked { background: #2563eb; color: white; font-weight: 600; }
#exitButton { background: #7f1d1d; color: white; border: none; border-radius: 8px; padding: 10px; }
#exitButton:hover { background: #991b1b; }
#thrValue { font-size: 22px; font-weight: 700; color: white; }
#card { background: white; border: 1px solid #e5e7eb; border-radius: 12px; }
#pageTitle { font-size: 26px; font-weight: 700; }
#pageSubtitle { font-size: 14px; color: #4b5563; }
#cardTitle { font-size: 16px; font-weight: 600; }
#fieldLabel { font-size: 14px; }
#hint { font-size: 12px; color: #6b7280; }
#metricTitle { font-size: 12px; color: #6b7280; }
#metricValue { font-size: 26px; font-weight: 700; }
#resultText { font-size: 40px; font-weight: 800; padding: 16px 0; }
#loadTitle { font-size: 30px; font-weight: 700; }
QPushButton { background: white; border: 1px solid #d1d5db; border-radius: 8px; padding: 9px 14px; }
QPushButton:hover { background: #f3f4f6; }
QPushButton#primary { background: #2563eb; color: white; border: none; font-weight: 600; }
QPushButton#primary:hover { background: #1d4ed8; }
QDoubleSpinBox { background: white; border: 1px solid #d1d5db; border-radius: 6px; padding: 6px 8px; }
QTableWidget { background: white; border: 1px solid #e5e7eb; gridline-color: #f3f4f6; }
QTableWidget::item:selected { background: #dbeafe; color: #111827; }
QHeaderView::section { background: #f9fafb; border: none; border-bottom: 1px solid #e5e7eb; padding: 6px; font-weight: 600; }
QProgressBar { background: #e5e7eb; border: none; border-radius: 6px; }
QProgressBar::chunk { background: #2563eb; border-radius: 6px; }
QSlider::groove:horizontal { height: 6px; background: #374151; border-radius: 3px; }
QSlider::handle:horizontal { background: #60a5fa; width: 18px; margin: -6px 0; border-radius: 9px; }
QSlider::sub-page:horizontal { background: #2563eb; border-radius: 3px; }
QToolTip { background: #111827; color: white; border: 1px solid #374151; padding: 6px; font-size: 13px; }
QScrollArea { background: transparent; }
"""

PAGE_NAMES = ["Prediction", "Performance", "Model Details", "Features"]


class MainWindow(QWidget):
    def __init__(self, data_path=DATA_PATH):
        super().__init__()
        self.setObjectName("appRoot")
        self.setWindowTitle("Breast Cancer Classification")
        self.setWindowFlags(Qt.FramelessWindowHint)  # kiosk mode
        self.data_path = data_path
        self.threshold = DEFAULT_THRESHOLD
        self.worker = None
        self.model = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.root = QStackedWidget()
        outer.addWidget(self.root)

        self.loading = LoadingPage()
        self.loading.choose_btn.clicked.connect(self.choose_file)
        self.loading.exit_btn.clicked.connect(QApplication.quit)
        self.root.addWidget(self.loading)

        QShortcut(QKeySequence("Ctrl+Q"), self, activated=QApplication.quit)
        self.start_training()

    # ---- training lifecycle
    def start_training(self):
        self.loading.show_busy()
        self.root.setCurrentWidget(self.loading)
        self.worker = TrainWorker(self.data_path)
        self.worker.done.connect(self.on_trained)
        self.worker.failed.connect(self.loading.show_error)
        self.worker.start()

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select dataset", ROOT_DIR, "CSV files (*.csv)")
        if path:
            self.data_path = path
            self.start_training()

    def on_trained(self, model):
        self.model = model
        self.workspace = self.build_workspace(model)
        self.root.addWidget(self.workspace)
        self.root.setCurrentWidget(self.workspace)

    # ---- workspace
    def build_workspace(self, model):
        ws = QWidget()
        lay = QHBoxLayout(ws)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # sidebar
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(270)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(18, 24, 18, 18)
        sl.setSpacing(6)
        brand = QLabel("Breast Cancer\nClassification")
        brand.setObjectName("brand")
        sub = QLabel("Logistic regression runner")
        sub.setObjectName("brandSub")
        sl.addWidget(brand)
        sl.addWidget(sub)
        sl.addSpacing(24)

        self.pages = QStackedWidget()
        self.prediction = PredictionPage(model, self.threshold)
        self.performance = PerformancePage(model, self.threshold)
        for page in (self.prediction, self.performance,
                     DetailsPage(model), FeaturesPage(model)):
            self.pages.addWidget(page)

        group = QButtonGroup(self)
        group.setExclusive(True)
        for i, name in enumerate(PAGE_NAMES):
            btn = QPushButton(name)
            btn.setObjectName("navButton")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, idx=i: self.pages.setCurrentIndex(idx))
            group.addButton(btn, i)
            sl.addWidget(btn)
            if i == 0:
                btn.setChecked(True)
        sl.addStretch(1)

        # global threshold control
        thr_title = QLabel("Decision threshold")
        thr_title.setToolTip(
            '<table width="300"><tr><td>A sample is classified as malignant when its '
            "predicted probability is at or above this value. Lower values catch more "
            "malignant cases but raise false alarms.</td></tr></table>")
        self.thr_value = QLabel("%.2f" % self.threshold)
        self.thr_value.setObjectName("thrValue")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(10, 90)
        slider.setSingleStep(5)
        slider.setPageStep(5)
        slider.setValue(int(round(self.threshold * 100)))
        slider.setToolTip(thr_title.toolTip())
        slider.valueChanged.connect(self.on_threshold)
        sl.addWidget(thr_title)
        sl.addWidget(self.thr_value)
        sl.addWidget(slider)
        sl.addSpacing(18)

        exit_btn = QPushButton("Exit application")
        exit_btn.setObjectName("exitButton")
        exit_btn.clicked.connect(QApplication.quit)
        sl.addWidget(exit_btn)

        lay.addWidget(sidebar)
        lay.addWidget(self.pages, 1)
        return ws

    def on_threshold(self, value):
        self.threshold = value / 100.0
        self.thr_value.setText("%.2f" % self.threshold)
        self.prediction.set_threshold(self.threshold)
        self.performance.set_threshold(self.threshold)


def main():
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.showFullScreen()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
