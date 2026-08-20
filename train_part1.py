import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance

# 1. Load Data
df = pd.read_csv("orders_dataset.csv")

# Missingness analysis verification
missing_cod = df[df["payment_method"] == "COD"]["rating_given"].isna().mean()
missing_non_cod = df[df["payment_method"] != "COD"]["rating_given"].isna().mean()

print("--- DATA SUMMARY ---")
print(f"Total Rows: {len(df)}")
print(f"Overall Return Rate: {df['returned'].mean():.4f}")
print(f"Rating Missing Rate (COD): {missing_cod:.4f} | Non-COD: {missing_non_cod:.4f}")
print("Missingness Type: MAR (Missing At Random), missingness depends on payment_method.\n")

# Features & Target
X = df.drop(columns=["order_id", "returned"])
y = df["returned"]

num_cols = ["price_inr", "discount_pct", "customer_tenure_days", "num_previous_orders", 
            "num_previous_returns", "delivery_distance_km", "delivery_days", 
            "is_weekend_order", "rating_given"]
cat_cols = ["product_category", "payment_method"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Pipeline Preprocessor
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

# 3. Dummy Baseline
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_preds = dummy.predict(X_test)
print("--- DUMMY BASELINE ---")
print(f"Accuracy: {accuracy_score(y_test, dummy_preds):.4f}")
print(f"F1 Score (Class 1): {f1_score(y_test, dummy_preds, zero_division=0):.4f}\n")

# 4. Logistic Regression & Threshold Sweep
lr_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(class_weight="balanced", random_state=42))
])
lr_pipe.fit(X_train, y_train)
lr_probs = lr_pipe.predict_proba(X_test)[:, 1]

print("--- LOGISTIC REGRESSION (Default 0.5 Threshold) ---")
print(f"ROC-AUC: {roc_auc_score(y_test, lr_probs):.4f}")
print(f"F1: {f1_score(y_test, lr_probs >= 0.5):.4f}")
print(f"Recall: {recall_score(y_test, lr_probs >= 0.5):.4f}")
print(f"Precision: {precision_score(y_test, lr_probs >= 0.5):.4f}\n")

best_lr_t, best_lr_f1 = 0.5, 0
for t in np.arange(0.1, 0.91, 0.02):
    f1 = f1_score(y_test, lr_probs >= t)
    if f1 > best_lr_f1:
        best_lr_f1 = f1
        best_lr_t = t

print(f"LR Optimal Threshold: {best_lr_t:.2f} | Max F1: {best_lr_f1:.4f}")
print(f"Recall at optimal: {recall_score(y_test, lr_probs >= best_lr_t):.4f} | Precision: {precision_score(y_test, lr_probs >= best_lr_t):.4f}\n")

# 5. Random Forest + GridSearchCV
rf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42))
])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [6, 10, None]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(rf_pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
rf_probs = best_rf.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, rf_probs)

print("--- RANDOM FOREST (GRID SEARCH WINNER) ---")
print(f"Best Params: {grid.best_params_}")
print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")
print(f"Test ROC-AUC: {test_auc:.4f}\n")
# -------------------------------------------------------------------
# Feature Importance Analysis & Permutation Importance Fix
# -------------------------------------------------------------------

# 1. Extract One-Hot Encoded feature names
ohe_feature_names = best_rf.named_steps["preprocessor"].named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(cat_cols)
feature_names = num_cols + list(ohe_feature_names)

# 2. Extract Impurity-based Feature Importances
importances = best_rf.named_steps["classifier"].feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "impurity_importance": importances}).sort_values("impurity_importance", ascending=False)

print("--- TOP 5 IMPURITY IMPORTANCES ---")
print(fi_df.head(5).to_string(index=False))

# 3. Permutation Importance (Transformed Data)
# Transform X_test through the preprocessor pipeline first so feature dimensions match
preprocessor_fitted = best_rf.named_steps["preprocessor"]
X_test_transformed = preprocessor_fitted.transform(X_test)
rf_classifier = best_rf.named_steps["classifier"]

perm_imp = permutation_importance(
    rf_classifier, 
    X_test_transformed, 
    y_test, 
    scoring="roc_auc", 
    n_repeats=10, 
    random_state=42
)

# Both arrays now have identical length (16 elements)
perm_df = pd.DataFrame({
    "feature": feature_names, 
    "perm_importance": perm_imp.importances_mean
}).sort_values("perm_importance", ascending=False)

print("\n--- TOP 5 PERMUTATION IMPORTANCES ---")
print(perm_df.head(5).to_string(index=False), "\n")

# Threshold Sweep for RF (t*_rf)
best_rf_t, best_rf_f1 = 0.5, 0
for t in np.arange(0.1, 0.91, 0.02):
    f1 = f1_score(y_test, rf_probs >= t)
    if f1 > best_rf_f1:
        best_rf_f1 = f1
        best_rf_t = t

print(f"Optimal t*_rf for Saved RF Model: {best_rf_t:.2f} (Max F1: {best_rf_f1:.4f})")

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save model artifact and meta threshold info
joblib.dump({"model": best_rf, "t_star_rf": round(float(best_rf_t), 2)}, "models/return_risk_model.pkl")
print("Saved RF Pipeline to models/return_risk_model.pkl")
