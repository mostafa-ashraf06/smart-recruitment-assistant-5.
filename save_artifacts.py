# -*- coding: utf-8 -*-
"""
save_artifacts.py — Smart Recruitment Assistant (Project 2)

Regenerates every deployment artifact from aug_train.csv via the exact notebook pipeline:
  - deterministic cleaning (no fitted statistics)
  - stratified 64/16/20 split (random_state=42) BEFORE any fitted statistic
  - training-only median imputation, row-wise feature engineering
  - train-fit encoding (city frequency / ordinal / one-hot) + StandardScaler (train-fit)
  - Logistic Regression / Random Forest / XGBoost (+ per-model F1 threshold from validation PR curve)
  - champion selected on VALIDATION AUPRC (test set never influences selection)

Outputs:
  recruitment_champion.pkl  — model, scaler, encodings, threshold, ranking pool (dashboard-ready)
  processed_dataset.csv     — cleaned + engineered human-readable dataset
  processed_candidates_encoded.csv — final 34 encoded features + split labels
  model_comparison.csv      — the three-model test table
  test_scores.csv           — champion scores on the untouched test set
"""
import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score,
)

RANDOM_STATE = 42
DATA_PATH = "aug_train.csv"
assert os.path.exists(DATA_PATH), "aug_train.csv must sit next to this script."

# ------------------------------------------------------------------
# 1) Load + deterministic cleaning (mirrors notebook Section 3)
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["experience"] = df["experience"].replace({"ERROR": np.nan, ">20": "21", "<1": "0"})
df["experience"] = pd.to_numeric(df["experience"])
df["last_new_job"] = df["last_new_job"].replace({"ERROR": np.nan, ">4": "5", "never": "0"})
df["last_new_job"] = pd.to_numeric(df["last_new_job"])
df["company_size"] = df["company_size"].replace("10/49", "10-49")
for col in ["gender", "education_level", "enrolled_university",
            "major_discipline", "company_size", "company_type"]:
    df[col] = df[col].fillna("UNKNOWN")

# ------------------------------------------------------------------
# 2) Split BEFORE any fitted statistic (mirrors notebook Section 4)
# ------------------------------------------------------------------
feature_cols = [c for c in df.columns if c not in ["enrollee_id", "target"]]
X = df[feature_cols]
y = df["target"]

X_raw_train_full, X_raw_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
X_raw_train, X_raw_val, y_train, y_val = train_test_split(
    X_raw_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_full)

experience_median = X_raw_train["experience"].median()
last_new_job_median = X_raw_train["last_new_job"].median()

# ------------------------------------------------------------------
# 3) Feature engineering (row-wise only)
# ------------------------------------------------------------------
def experience_bucket(x):
    if x <= 3:
        return 0
    elif x <= 9:
        return 1
    return 2

def engineer(frame, exp_median, lnj_median):
    out = frame.copy()
    out["experience"] = out["experience"].fillna(exp_median)
    out["last_new_job"] = out["last_new_job"].fillna(lnj_median)
    out["experience_level"] = out["experience"].apply(experience_bucket)
    out["is_recent_mover"] = ((out["last_new_job"] <= 1) & (out["experience"] > 1)).astype(int)
    out["training_per_experience"] = out["training_hours"] / (out["experience"] + 1)
    out["experience_x_cdi"] = out["experience"] * out["city_development_index"]
    return out

X_train = engineer(X_raw_train, experience_median, last_new_job_median)
X_val = engineer(X_raw_val, experience_median, last_new_job_median)
X_test = engineer(X_raw_test, experience_median, last_new_job_median)

# ------------------------------------------------------------------
# 4) Train-fit encoding + scaling (mirrors notebook Section 6)
# ------------------------------------------------------------------
city_freq = X_train["city"].value_counts(normalize=True)
ORDINAL_EDU = {"UNKNOWN": 0, "Primary School": 1, "High School": 2, "Graduate": 3, "Masters": 4, "Phd": 5}
ORDINAL_SIZE = {"UNKNOWN": 0, "<10": 1, "10-49": 2, "50-99": 3, "100-500": 4,
                "500-999": 5, "1000-4999": 6, "5000-9999": 7, "10000+": 8}
ONEHOT_COLS = ["gender", "enrolled_university", "major_discipline", "company_type"]

def encode_features(X_part):
    out = X_part.copy()
    out["city_frequency"] = out["city"].map(city_freq).fillna(0.0)
    out = out.drop(columns=["city"])
    out["relevent_experience"] = out["relevent_experience"].map(
        {"Has relevent experience": 1, "No relevent experience": 0})
    out["education_level"] = out["education_level"].map(ORDINAL_EDU).fillna(0)
    out["company_size"] = out["company_size"].map(ORDINAL_SIZE).fillna(0)
    out = pd.get_dummies(out, columns=ONEHOT_COLS, dtype=int)
    return out

X_train_enc = encode_features(X_train)
feature_names = X_train_enc.columns.tolist()
X_val_enc = encode_features(X_val).reindex(columns=feature_names, fill_value=0)
X_test_enc = encode_features(X_test).reindex(columns=feature_names, fill_value=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enc)
X_val_scaled = scaler.transform(X_val_enc)
X_test_scaled = scaler.transform(X_test_enc)
X_train_arr, X_val_arr, X_test_arr = (f.to_numpy() for f in (X_train_enc, X_val_enc, X_test_enc))

# ------------------------------------------------------------------
# 5) Models + per-model F1 threshold from the validation PR curve
# ------------------------------------------------------------------
def best_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])

spw = float((y_train == 0).sum() / (y_train == 1).sum())
model_specs = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": GridSearchCV(
        RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                               random_state=RANDOM_STATE, n_jobs=-1),
        param_grid={"max_depth": [10, 14, None], "min_samples_leaf": [1, 5]},
        scoring="roc_auc", cv=3, n_jobs=-1),
    "XGBoost": GridSearchCV(
        XGBClassifier(n_estimators=300, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
                      min_child_weight=5, tree_method="hist", eval_metric="logloss",
                      scale_pos_weight=spw, random_state=RANDOM_STATE, n_jobs=-1),
        param_grid={"max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1]},
        scoring="roc_auc", cv=3, n_jobs=1),
}

results_rows, results = [], {}
for name, obj in model_specs.items():
    needs_scaling = name == "Logistic Regression"
    if isinstance(obj, GridSearchCV):
        obj.fit(X_train_arr if not needs_scaling else X_train_scaled, y_train)
        model = obj.best_estimator_
    else:
        obj.fit(X_train_scaled if needs_scaling else X_train_arr, y_train)
        model = obj
    val_proba = model.predict_proba(X_val_scaled if needs_scaling else X_val_arr)[:, 1]
    threshold, val_f1 = best_threshold(y_val, val_proba)

    test_proba = model.predict_proba(X_test_scaled if needs_scaling else X_test_arr)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    row = {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, test_pred),
        "Precision": precision_score(y_test, test_pred),
        "Recall": recall_score(y_test, test_pred),
        "F1 Score": f1_score(y_test, test_pred),
        "ROC AUC": roc_auc_score(y_test, test_proba),
        "Val AUPRC": average_precision_score(y_val, val_proba),
    }
    print(f"{name:22s} thr={threshold:.3f}  acc={row['Accuracy']:.3f}  prec={row['Precision']:.3f} "
          f"rec={row['Recall']:.3f}  f1={row['F1 Score']:.3f}  auc={row['ROC AUC']:.3f}  "
          f"valAP={row['Val AUPRC']:.3f}")
    results_rows.append(row)
    results[name] = dict(model=model, threshold=threshold, proba=test_proba, pred=test_pred,
                         val_ap=row["Val AUPRC"], needs_scaling=needs_scaling)

# Champion: highest validation AUPRC — the test set never influences this choice.
champion_name = max(results, key=lambda n: results[n]["val_ap"])
champion = results[champion_name]
print(f"\nChampion (by validation AUPRC): {champion_name}")

# ------------------------------------------------------------------
# 6) Artifacts
# ------------------------------------------------------------------
model_comparison = pd.DataFrame(results_rows).sort_values("Val AUPRC", ascending=False).reset_index(drop=True)

df_processed = pd.concat([
    engineer(X_raw_train_full, experience_median, last_new_job_median),
    engineer(X_raw_test, experience_median, last_new_job_median),
]).sort_index()
df_processed.insert(0, "enrollee_id", df.loc[df_processed.index, "enrollee_id"])
df_processed["target"] = y.loc[df_processed.index]
df_processed["split"] = np.where(df_processed.index.isin(X_raw_val.index), "validation",
                        np.where(df_processed.index.isin(X_raw_test.index), "test", "train"))

X_all_enc = pd.concat([X_train_enc, X_val_enc, X_test_enc]).sort_index()
enc_dataset = X_all_enc.copy()
enc_dataset.insert(0, "enrollee_id", df.loc[enc_dataset.index, "enrollee_id"])
enc_dataset["target"] = y.loc[enc_dataset.index]
enc_dataset["split"] = np.where(enc_dataset.index.isin(X_test_enc.index), "test",
                        np.where(enc_dataset.index.isin(X_val_enc.index), "validation", "train"))

test_scores = pd.DataFrame({
    "enrollee_id": df.loc[X_test.index, "enrollee_id"].values,
    "predicted_probability": champion["proba"],
    "actually_looking": y_test.values,
}).sort_values("predicted_probability", ascending=False).reset_index(drop=True)

bundle = {
    "model": champion["model"],
    "model_name": champion_name,
    "needs_scaling": champion["needs_scaling"],
    "scaler": scaler,
    "feature_names": feature_names,
    "threshold": champion["threshold"],
    "city_freq": city_freq,
    "imputation": {"experience_median": experience_median,
                   "last_new_job_median": last_new_job_median},
    "trained_on": f"aug_train.csv ({df.shape[0]:,} rows) — stratified 64/16/20 split, random_state={RANDOM_STATE}",
    "val_auprc": champion["val_ap"],
    "test_metrics": model_comparison.set_index("Model").loc[champion_name].to_dict(),
    "model_comparison": model_comparison,
    "test_scores": test_scores,
}

joblib.dump(bundle, "recruitment_champion.pkl")
model_comparison.to_csv("model_comparison.csv", index=False)
test_scores.to_csv("test_scores.csv", index=False)
df_processed.reset_index(drop=True).to_csv("processed_dataset.csv", index=False)
enc_dataset.reset_index(drop=True).to_csv("processed_candidates_encoded.csv", index=False)

print("\nSaved artifacts (bit-reproducible with random_state=42):")
for f in ["recruitment_champion.pkl", "processed_dataset.csv",
          "processed_candidates_encoded.csv", "model_comparison.csv", "test_scores.csv"]:
    print(f"  {f:<38} {os.path.getsize(f) / 1024:>8.1f} KB")

cm = confusion_matrix(y_test, champion["pred"])
print(f"\nChampion test confusion matrix [TN FP / FN TP]:\n{cm}")
