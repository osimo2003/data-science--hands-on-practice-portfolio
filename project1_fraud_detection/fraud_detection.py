# ============================================
# PROJECT 1 — CREDIT CARD FRAUD DETECTION
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# --- Load the dataset ---
df = pd.read_csv('creditcard.csv')

# --- First look at the data ---
print("=== DATASET SHAPE ===")
print("Rows and Columns:", df.shape)

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== DATA TYPES & MISSING VALUES ===")
print(df.info())

print("\n=== BASIC STATISTICS ===")
print(df.describe())

print("\n=== CLASS DISTRIBUTION ===")
print(df['Class'].value_counts())

print("\nFraud Percentage:")
print(df['Class'].value_counts(normalize=True) * 100)

# ============================================
# BLOCK 2 — DATA PREPROCESSING & VISUALISATION
# ============================================

# --- Visualise the class imbalance ---
plt.figure(figsize=(8, 5))
sns.countplot(x='Class', data=df, palette='Set2')
plt.title('Class Distribution - Normal vs Fraud')
plt.xlabel('Class (0 = Normal, 1 = Fraud)')
plt.ylabel('Number of Transactions')
plt.savefig('class_distribution.png')
print("\n=== CLASS DISTRIBUTION CHART SAVED ===")

# --- Check the transaction amount for fraud vs normal ---
print("\n=== AVERAGE AMOUNT: NORMAL vs FRAUD ===")
print(df.groupby('Class')['Amount'].describe())

# --- Scale the Amount and Time columns ---
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# --- Drop the original unscaled columns ---
df.drop(['Amount', 'Time'], axis=1, inplace=True)

print("\n=== SCALING COMPLETE ===")
print("New shape after scaling:", df.shape)
print("\nFirst 5 rows after scaling:")
print(df.head())

# ============================================
# BLOCK 3 — SPLIT DATA & FIX IMBALANCE WITH SMOTE
# ============================================

# --- Separate features (X) and target (y) ---
X = df.drop('Class', axis=1)
y = df['Class']

print("\n=== FEATURES AND TARGET ===")
print("X shape:", X.shape)
print("y shape:", y.shape)

# --- Split into training and testing sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n=== TRAIN & TEST SPLIT ===")
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

print("\nClass distribution in training set BEFORE SMOTE:")
print(y_train.value_counts())

# --- Apply SMOTE to fix the imbalance ---
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nClass distribution in training set AFTER SMOTE:")
print(pd.Series(y_train_smote).value_counts())
print("\nSMOTE applied successfully!")

# ============================================
# BLOCK 4 — BUILD & TRAIN ML MODELS
# ============================================

# --- MODEL 1: Random Forest ---
print("\n=== TRAINING RANDOM FOREST MODEL ===")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_smote, y_train_smote)
rf_predictions = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest Training Complete!")

# --- MODEL 2: XGBoost ---
print("\n=== TRAINING XGBOOST MODEL ===")
xgb_model = XGBClassifier(n_estimators=100, random_state=42, 
                           eval_metric='logloss', n_jobs=-1)
xgb_model.fit(X_train_smote, y_train_smote)
xgb_predictions = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

print("XGBoost Training Complete!")

# ============================================
# BLOCK 5 — EVALUATE BOTH MODELS
# ============================================

# --- Random Forest Evaluation ---
print("\n=== RANDOM FOREST RESULTS ===")
print(classification_report(y_test, rf_predictions))
print("ROC-AUC Score:", roc_auc_score(y_test, rf_proba))

# --- Confusion Matrix for Random Forest ---
rf_cm = confusion_matrix(y_test, rf_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
plt.title('Random Forest - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('rf_confusion_matrix.png')
print("Random Forest Confusion Matrix saved!")

# --- XGBoost Evaluation ---
print("\n=== XGBOOST RESULTS ===")
print(classification_report(y_test, xgb_predictions))
print("ROC-AUC Score:", roc_auc_score(y_test, xgb_proba))

# --- Confusion Matrix for XGBoost ---
xgb_cm = confusion_matrix(y_test, xgb_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
plt.title('XGBoost - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('xgb_confusion_matrix.png')
print("XGBoost Confusion Matrix saved!")

# --- Compare Botsh Models ---
print("\n=== MODEL COMPARISON ===")
print(f"Random Forest ROC-AUC: {roc_auc_score(y_test, rf_proba):.4f}")
print(f"XGBoost ROC-AUC:       {roc_auc_score(y_test, xgb_proba):.4f}")

# ============================================
# BLOCK 6 — SAVE THE MODELS
# ============================================

import joblib

# --- Save Random Forest Model ---
joblib.dump(rf_model, 'random_forest_model.pkl')
print("\n=== MODELS SAVED ===")
print("Random Forest model saved as: random_forest_model.pkl")

# --- Save XGBoost Model ---
joblib.dump(xgb_model, 'xgboost_model.pkl')
print("XGBoost model saved as: xgboost_model.pkl")

# --- Save the Scaler ---
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as: scaler.pkl")

# --- Confirm all saved files ---
import os
print("\n=== FILES IN PROJECT FOLDER ===")
for file in os.listdir('.'):
    size = os.path.getsize(file)
    print(f"{file}  —  {size / 1024:.1f} KB")

print("\n PROJECT 1 COMPLETE!")
print("Credit Card Fraud Detection model built and saved successfully!")