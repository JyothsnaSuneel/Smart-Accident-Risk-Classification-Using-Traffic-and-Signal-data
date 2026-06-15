# test_model_loader.py

import sys
sys.path.append(r"C:\Users\sunee\SmartAccidentStreamlitApp\utils")

from model_loader import load_ensemble_model, load_feature_names

model = load_ensemble_model()
features = load_feature_names()
print("✅ Model and features loaded successfully!")


