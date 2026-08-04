"""
Train the Travel Safety Advisor model.

Fixes applied vs. the original save_model.py:
  1. Removed data leakage: 'cluster' and 'safe_or_unsafe' (string) were
     literally derived from the target and were being fed in as features.
     The old model wasn't predicting anything -- it was reading the answer
     off two leaked columns.
  2. Dropped 'Deaths' (71% missing -- imputing it just injects noise) and
     'Disease'/'Unnamed: 0' (not something a traveler would ever know or
     need before a trip).
  3. Feature list here now exactly matches what the Streamlit app collects
     from the user, so nothing gets silently zero-filled at prediction time.
  4. Categorical encoders now map unseen labels to a dedicated "unknown"
     bucket instead of crashing or silently defaulting to 0.
  5. Proper train/test split + stratification + classification report so
     you can see real, honest accuracy instead of leaked ~100% accuracy.
  6. class_weight='balanced' because the dataset is ~80/20 safe/unsafe.

Run:
    python train_model.py
Produces (all under ./saved/):
    random_forest_model.joblib, scaler.joblib, imputer.joblib,
    feature_names.joblib, encoders.joblib, metrics.txt
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

DATA_PATH = Path(__file__).parent / "health_crisis_with_safety.csv"
OUT_DIR = Path(__file__).parent / "saved"
OUT_DIR.mkdir(exist_ok=True)

TARGET = "safe_or_unsafe_binary"

# Columns that leak the target or are not knowable before a trip.
DROP_COLS = ["Unnamed: 0", "cluster", "safe_or_unsafe", "Deaths", "Disease"]

CATEGORICAL_COLS = ["week_of_outbreak", "state_ut", "district"]

print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# Cases arrives as text with occasional junk -> coerce to numeric.
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")

y = df[TARGET]
X = df.drop(columns=[TARGET])

print(f"Rows: {len(df)}  |  Features: {X.shape[1]}  |  Target balance:\n{y.value_counts(normalize=True)}")

# ---- Encode categoricals with an explicit "unknown" bucket ----
encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    values = X[col].astype(str).tolist() + ["__UNKNOWN__"]
    le.fit(values)
    X[col] = le.transform(X[col].astype(str))
    encoders[col] = le

feature_names = X.columns.tolist()

# ---- Split BEFORE imputing/scaling to avoid any test-set leakage ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

print("Training RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_scaled, y_train)

# ---- Honest evaluation on held-out data ----
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

report = classification_report(y_test, y_pred, target_names=["unsafe", "safe"])
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(report)
print(f"Accuracy: {acc:.3f}  |  ROC-AUC: {auc:.3f}")

importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nTop feature importances:")
print(importances.head(10))

with open(OUT_DIR / "metrics.txt", "w") as f:
    f.write("Held-out test set (20% of data, stratified split)\n\n")
    f.write(report)
    f.write(f"\nAccuracy: {acc:.3f}\nROC-AUC: {auc:.3f}\n\n")
    f.write("Top feature importances:\n")
    f.write(importances.head(15).to_string())

# ---- Refit on ALL data for the deployed model (best practice: more data = better) ----
X_all_imp = imputer.transform(X)  # imputer/scaler already fit on train only is fine to reuse
X_all_scaled = scaler.transform(X_all_imp)
model.fit(X_all_scaled, y)

joblib.dump(model, OUT_DIR / "random_forest_model.joblib")
joblib.dump(scaler, OUT_DIR / "scaler.joblib")
joblib.dump(imputer, OUT_DIR / "imputer.joblib")
joblib.dump(feature_names, OUT_DIR / "feature_names.joblib")
joblib.dump(encoders, OUT_DIR / "encoders.joblib")

# Save dropdown choices for the app (real state/district names, sorted)
joblib.dump(sorted(df["state_ut"].unique().tolist()), OUT_DIR / "state_options.joblib")
district_by_state = df.groupby("state_ut")["district"].unique().apply(lambda a: sorted(a.tolist())).to_dict()
joblib.dump(district_by_state, OUT_DIR / "district_by_state.joblib")

print(f"\nSaved model + artifacts to {OUT_DIR}/")
