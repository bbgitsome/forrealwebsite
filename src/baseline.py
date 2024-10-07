import numpy as np
import librosa
import xgboost as xgb
from spafe.features.gfcc import gfcc
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_baseline_model():
    # Load the model
    xgb_model = xgb.XGBClassifier()
    model_path = "models/baseline/BASELINE_XGBoostCLEAN_Optuna.json"
    xgb_model.load_model(model_path)
    return  xgb_model

def extract_handcrafted_features(audio, sr=16000):
    cqt = np.abs(librosa.cqt(audio, sr=sr, n_bins=84, bins_per_octave=12, fmin=librosa.note_to_hz('C1')))
    cqt_features = np.hstack([
        np.mean(cqt, axis=1),
        np.std(cqt, axis=1),
        np.max(cqt, axis=1),
        np.min(cqt, axis=1)
    ])

    gfcc_features = gfcc(audio, fs=sr, num_ceps=20, nfilts=40)
    gfcc_features = np.hstack([
        np.mean(gfcc_features, axis=0),
        np.std(gfcc_features, axis=0),
        np.max(gfcc_features, axis=0),
        np.min(gfcc_features, axis=0)
    ])
    return cqt_features, gfcc_features

def predict_baseline(audio_path, xgb_model):
    audio, sr = librosa.load(audio_path, sr=16000, duration=60)

    cqt_features, gfcc_features = extract_handcrafted_features(audio)

    combined_features = np.concatenate([
        cqt_features,
        gfcc_features
    ]).reshape(1, -1)

    prediction_proba = xgb_model.predict_proba(combined_features)[0]
    spoof_probability = prediction_proba[1]
    prediction = 1 if spoof_probability > 0.5 else 0

    feature_importances = xgb_model.feature_importances_
    
    return prediction, float(spoof_probability)
