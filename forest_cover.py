import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Forest Cover Type Classification
# Compare Random Forest vs. XGBoost + Hyperparameter Tuning
# This script loads the Forest Cover Type dataset, trains both Random Forest and XGBoost classifiers,
# evaluates their performance, and performs hyperparameter tuning on the Random Forest model.
# 1. Load Dataset
print("Loading dataset...")
data = fetch_covtype(as_frame=True)
df = data.frame

# Separate features and target
X = df.drop(columns=["Cover_Type"])
y = df["Cover_Type"]

# ⚠️ FIX: Adjust labels for XGBoost (should start from 0)
y_xgb = y - 1  

print(f"Dataset shape: {df.shape}")
print(f"Unique cover types: {y.unique()}")

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Also split for XGBoost with shifted labels
_, _, y_train_xgb, y_test_xgb = train_test_split(
    X, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb
)

# 3. Train Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {acc_rf:.4f}")

# 4. Train XGBoost
print("\nTraining XGBoost...")
xgb = XGBClassifier(
    n_estimators=200, max_depth=8, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, random_state=42, eval_metric="mlogloss"
)
xgb.fit(X_train, y_train_xgb)  # train on shifted labels
y_pred_xgb = xgb.predict(X_test)
acc_xgb = accuracy_score(y_test_xgb, y_pred_xgb)  # compare on shifted labels
print(f"XGBoost Accuracy: {acc_xgb:.4f}")

# 5. Model Comparison Reports
print("\nClassification Reports:\n")
print("Random Forest:\n", classification_report(y_test, y_pred_rf))
print("XGBoost:\n", classification_report(y_test_xgb, y_pred_xgb))

# 6. Confusion Matrices (saved as images)
os.makedirs("images", exist_ok=True)

def plot_conf_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"images/confusion_matrix_{model_name}.png")
    plt.close()

plot_conf_matrix(y_test, y_pred_rf, "RandomForest")
plot_conf_matrix(y_test_xgb, y_pred_xgb, "XGBoost")

# 7. Feature Importance Plots
def plot_feature_importance(model, model_name, feature_names, top_n=15):
    importances = model.feature_importances_
    idx = importances.argsort()[-top_n:][::-1]
    plt.figure(figsize=(10,6))
    sns.barplot(x=importances[idx], y=[feature_names[i] for i in idx])
    plt.title(f"Top {top_n} Feature Importances - {model_name}")
    plt.savefig(f"images/feature_importance_{model_name}.png")
    plt.close()

plot_feature_importance(rf, "RandomForest", X.columns)
plot_feature_importance(xgb, "XGBoost", X.columns)

# 8. Hyperparameter Tuning (Random Forest)
print("\nHyperparameter tuning Random Forest...")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5]
}

grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid,
                       cv=3, scoring="accuracy", verbose=1, n_jobs=-1)
grid_rf.fit(X_train, y_train)

print(f"Best RF Params: {grid_rf.best_params_}")
print(f"Best RF Accuracy (CV): {grid_rf.best_score_:.4f}")
