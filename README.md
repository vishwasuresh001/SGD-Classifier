# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the data file and import numpy, matplotlib and scipy.
2. 2.Visulaize the data and define the sigmoid function, cost function and gradient descent.
3. 3.Plot the decision boundary .
4.  4.Calculate the y-prediction.


## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: VISHWA S
RegisterNumber:212225040495
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from sklearn.calibration import CalibratedClassifierCV

import warnings
warnings.filterwarnings("ignore")


# -------------------------
# 1. Load dataset
# -------------------------

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset:", data.DESCR.splitlines()[0])
print("X shape:", X.shape, "y shape:", y.shape)
print()


# -------------------------
# 2. Quick EDA
# -------------------------

print("First 5 rows:")
print(X.iloc[:, :6].head())


# -------------------------
# 3. Select 2 features
#    for visualization
# -------------------------

feat1, feat2 = "mean radius", "mean texture"

X_vis = X[[feat1, feat2]].values

# Full features for model training
X_full = X.values


# -------------------------
# 4. Train-test split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_full,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

Xv_train, Xv_test, yv_train, yv_test = train_test_split(
    X_vis,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# -------------------------
# 5. Scaling
# -------------------------

scaler = StandardScaler().fit(X_train)

X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

scaler_vis = StandardScaler().fit(Xv_train)

Xv_train_s = scaler_vis.transform(Xv_train)
Xv_test_s = scaler_vis.transform(Xv_test)


# -------------------------
# 6. Train SGDClassifier
#    as Logistic Regression
# -------------------------

clf = SGDClassifier(
    loss='log_loss',
    penalty='l2',
    alpha=1e-4,
    max_iter=1000,
    tol=1e-4,
    learning_rate='optimal',
    random_state=42,
    verbose=0
)

clf.fit(X_train_s, y_train)


# SGDClassifier does not provide
# predict_proba by default.
# Calibration is used to obtain probabilities.

calibrated = CalibratedClassifierCV(
    clf,
    method="sigmoid",
    cv=5
)

calibrated.fit(X_train_s, y_train)


# -------------------------
# 7. Evaluation
# -------------------------

y_pred = clf.predict(X_test_s)

y_proba = calibrated.predict_proba(X_test_s)[:, 1]


acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("Test set metrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print()


# -------------------------
# 8. Confusion Matrix
# -------------------------

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
).plot()

plt.title("Confusion Matrix")
plt.show()


# -------------------------
# 9. ROC Curve
# -------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

roc_auc = auc(fpr, tpr)

print(f"ROC-AUC: {roc_auc:.4f}")

plt.figure(figsize=(6, 4))

plt.plot(
    fpr,
    tpr,
    label=f"ROC curve (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
# --------------------
# 7. Evaluation on test set
# --------------------
y_pred = clf.predict(X_test_s)
y_proba = calibrated.predict_proba(X_test_s)[:, 1]  # probability of class 1

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Test set metrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print()
 Cross-validation (optional)
cv_scores = cross_val_score(clf, scaler.transform(X_full), y, cv=5, scoring='accuracy')
print("5-fold CV accuracy: mean={:.4f} std={:.4f}".format(cv_scores.mean(), cv_scores.std()))
print()
# --------------------
# 8. Confusion matrix
# --------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
fig, ax = plt.subplots(figsize=(5,4))
disp.plot(ax=ax)
ax.set_title("Confusion Matrix (Test set)")
plt.show()
# --------------------
# 9. ROC curve
# --------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(6,5))
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="SGD Logistic (calibrated)").plot(ax=ax)
ax.set_title(f"ROC Curve (AUC = {roc_auc:.3f})")
plt.show()

# --------------------
# 10. Decision boundary plot (2 features) for visual learners
#     We'll train a separate SGD model on just the 2 features.
# --------------------
clf_vis = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-4, random_state=42)
clf_vis.fit(Xv_train_s, yv_train)

# Make mesh
xx_min, xx_max = Xv_train_s[:, 0].min() - 1, Xv_train_s[:, 0].max() + 1
yy_min, yy_max = Xv_train_s[:, 1].min() - 1, Xv_train_s[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(xx_min, xx_max, 300), np.linspace(yy_min, yy_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

Z = clf_vis.predict(grid).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7,6))
ax.contourf(xx, yy, Z, alpha=0.2)
# plot training points
ax.scatter(Xv_train_s[:, 0][yv_train==0], Xv_train_s[:, 1][yv_train==0], marker='o', label=data.target_names[0], edgecolor='k')
ax.scatter(Xv_train_s[:, 0][yv_train==1], Xv_train_s[:, 1][yv_train==1], marker='^', label=data.target_names[1], edgecolor='k')
ax.set_xlabel(feat1 + " (scaled)")
ax.set_ylabel(feat2 + " (scaled)")
ax.set_title("Decision boundary (SGD Logistic) — trained on 2 features")
ax.legend()
plt.show()

*/
```

## Output:
<img width="613" height="502" alt="image" src="https://github.com/user-attachments/assets/6bc4e3fe-e2c9-48e5-830d-2d2132f5febd" />

<img width="593" height="596" alt="image" src="https://github.com/user-attachments/assets/79d955e5-cfc8-4682-bbfb-080f0b0af369" />

<img width="775" height="697" alt="image" src="https://github.com/user-attachments/assets/6d02f90d-d71e-459e-9d9a-4daf6ffcb6f6" />

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
