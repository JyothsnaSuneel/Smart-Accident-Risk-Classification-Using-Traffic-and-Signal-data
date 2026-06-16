# Final Ensemble Code: CatBoost + LightGBM + XGBoost with SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from imblearn.combine import SMOTEENN
import pandas as pd
import os
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\sunee\OneDrive\Desktop\360digitmg_projects\Smart Accident Risk Classification using Traffic and Signal Data\Smart_accident_Project_dataset_provided.csv")
df.drop(columns=['location_id'], errors='ignore', inplace=True)
df.dropna(thresh=0.1 * len(df), axis=1, inplace=True)
df = df.dropna(subset=['accident_occurred'])

# Fill and encode
for col in df.select_dtypes(include='number'):
    df[col].fillna(df[col].median(), inplace=True)
encoders = {}

for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0], inplace=True)

    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    
encoders[col] = le

X = df.drop('accident_occurred', axis=1)
y = df['accident_occurred']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# Balance data
smoteenn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smoteenn.fit_resample(X_train, y_train)

# Models
cat = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                         border_count=128, verbose=0, random_state=42)
lgb = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=10,
                     min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                     class_weight='balanced', random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                    learning_rate=0.05, max_depth=5, n_estimators=200, random_state=42)

# Voting Ensemble
ensemble = VotingClassifier(estimators=[
    ('catboost', cat),
    ('lightgbm', lgb),
    ('xgboost', xgb)
], voting='soft')

# Train
ensemble.fit(X_train_res, y_train_res)

# Evaluate
y_pred = ensemble.predict(X_test)
y_proba = ensemble.predict_proba(X_test)[:, 1]
print("\n✅ Ensemble Evaluation")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("Accuracy:", accuracy_score(y_test, y_pred))
import os
import joblib

# Absolute path to your project folder
project_path = r"C:\Users\sunee\SmartAccidentStreamlitApp"
model_folder = os.path.join(project_path, "models")
os.makedirs(model_folder, exist_ok=True)

# Save encoders
joblib.dump(encoders, os.path.join(model_folder, "label_encoders.pkl"))
print("✅ label_encoders.pkl saved")

# Save full path
feature_path = os.path.join(model_folder, "feature_names.pkl")
joblib.dump(X.columns.tolist(), feature_path)

print(f"✅ feature_names.pkl saved at: {feature_path}")

ensemble_path = os.path.join(model_folder, "ensemble_model.pkl")
joblib.dump(ensemble, ensemble_path)

print(f"✅ ensemble_model.pkl saved at: {ensemble_path}")

