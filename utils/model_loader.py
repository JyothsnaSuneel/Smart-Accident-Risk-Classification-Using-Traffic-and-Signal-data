
# utils/model_loader.py
import joblib
import os
import sys

# Fallback for __file__ in notebooks or Spyder
try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()  # fallback for interactive environments

MODELS_DIR = os.path.join(CURRENT_DIR, '..', 'models')

def load_ensemble_model():
    model_path = os.path.join(MODELS_DIR, 'ensemble_model.pkl')
    return joblib.load(model_path)

def load_feature_names():
    feature_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
    return joblib.load(feature_path)





