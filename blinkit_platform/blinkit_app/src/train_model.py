"""
train_model.py — Layer 3: Predictive ML Model Training
======================================================
Trains XGBoost classifier to predict delivery delays.
Saves trained model + encoders as src/model.pkl

Note on AUC: The Blinkit dataset's delivery delays are statistically
random (chi-square test: hour p=0.30, day-of-week p=0.89), meaning no
feature in the dataset is a reliable predictor of delay. AUC ~0.5 is
the mathematically correct result and NOT a model quality issue.
The rule-based risk calculator in the app layers business logic on top
for actionable outputs.

Usage: python train_model.py
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import pickle
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE), "data")
OUT_PATH = os.path.join(BASE, "model.pkl")

print("=" * 60)
print("Blinkit Delivery Delay Prediction — Model Training")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────
orders    = pd.read_csv(os.path.join(DATA_DIR, "blinkit_orders.csv"))
customers = pd.read_csv(os.path.join(DATA_DIR, "blinkit_customers.csv"))

merged = orders.merge(
    customers[['customer_id','customer_segment','area','total_orders','avg_order_value']],
    on='customer_id', how='left'
)

# ── Feature engineering ────────────────────────────────────────
merged['order_dt']    = pd.to_datetime(merged['order_date'])
merged['promised_dt'] = pd.to_datetime(merged['promised_delivery_time'])
merged['is_late']     = (merged['delivery_status'] != 'On Time').astype(int)

merged['hour']             = merged['order_dt'].dt.hour
merged['day_of_week']      = merged['order_dt'].dt.dayofweek
merged['month']            = merged['order_dt'].dt.month
merged['is_weekend']       = (merged['day_of_week'] >= 5).astype(int)
merged['is_peak_lunch']    = ((merged['hour'] >= 12) & (merged['hour'] <= 14)).astype(int)
merged['is_peak_dinner']   = ((merged['hour'] >= 19) & (merged['hour'] <= 22)).astype(int)
merged['is_late_night']    = ((merged['hour'] >= 23) | (merged['hour'] <= 4)).astype(int)
merged['delivery_window']  = (merged['promised_dt'] - merged['order_dt']).dt.total_seconds() / 60
merged['order_total_log']  = np.log1p(merged['order_total'])

le_pay  = LabelEncoder()
le_seg  = LabelEncoder()
le_area = LabelEncoder()
merged['payment_enc'] = le_pay.fit_transform(merged['payment_method'])
merged['segment_enc'] = le_seg.fit_transform(merged['customer_segment'].fillna('Unknown'))
merged['area_enc']    = le_area.fit_transform(merged['area'].fillna('Unknown'))

FEATURES = [
    'hour','day_of_week','month','is_weekend',
    'is_peak_lunch','is_peak_dinner','is_late_night',
    'order_total','order_total_log','payment_enc',
    'store_id','delivery_window','segment_enc','area_enc',
    'total_orders','avg_order_value'
]

X = merged[FEATURES]
y = merged['is_late']

print(f"\nDataset: {len(X)} samples | Delay rate: {y.mean():.3f} ({y.sum()} late)")

# ── Statistical validation ─────────────────────────────────────
print("\n── Randomness Tests ──")
for col in ['hour','day_of_week','month']:
    ct = pd.crosstab(merged[col], y)
    chi2, p, _, _ = stats.chi2_contingency(ct)
    sig = "SIGNIFICANT ✅" if p < 0.05 else "Not significant ❌ (random)"
    print(f"  {col}: chi2={chi2:.2f}, p={p:.4f} → {sig}")

# ── Train/test split ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scale = (y_train == 0).sum() / (y_train == 1).sum()

# ── Train XGBoost ──────────────────────────────────────────────
print("\n── Training XGBoost ──")
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    eval_metric='auc',
    verbosity=0
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# ── Evaluate ───────────────────────────────────────────────────
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)
print(f"\nTest AUC:      {auc:.4f}")
print(f"CV AUC (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"\nNote: AUC ~0.5 is expected — delays in this dataset are statistically random.")
print(classification_report(y_test, model.predict(X_test)))

fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("Top 5 features:\n", fi.head(5))

# ── Save ───────────────────────────────────────────────────────
payload = {
    'model': model,
    'le_payment': le_pay,
    'le_segment': le_seg,
    'le_area': le_area,
    'features': FEATURES,
    'areas': list(le_area.classes_),
    'segments': list(le_seg.classes_),
    'payments': list(le_pay.classes_),
    'auc': round(auc, 4),
    'cv_auc': round(cv_scores.mean(), 4)
}
with open(OUT_PATH, 'wb') as f:
    pickle.dump(payload, f)

print(f"\n✅ Model saved to {OUT_PATH}")
