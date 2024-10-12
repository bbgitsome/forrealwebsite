import joblib
import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import AutoProcessor, WavLMModel
from spafe.features.gfcc import gfcc

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states).squeeze(-1)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        attended_hidden = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        return attended_hidden

class CustomWavLMForFeatureExtraction(nn.Module):
    def __init__(self, model_name):
        super(CustomWavLMForFeatureExtraction, self).__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.attention = AttentionLayer(self.wavlm.config.hidden_size)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        attended_hidden = self.attention(hidden_states)
        return attended_hidden

best_threshold = 0.5700

def load_for_real_model():

    model_data = joblib.load('models/for_real/weighted.joblib')
    calibrated_model = model_data['calibrated_model']
    selector = model_data['selector']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
    processor = AutoProcessor.from_pretrained(model_name)

    wavlm_model = CustomWavLMForFeatureExtraction(model_name).to(device)

    state_dict = torch.load('models/for_real/best_wavlm_asvspoof_model.pth', map_location=device)
    keys_to_remove = ["classifier.weight", "classifier.bias"]
    for key in keys_to_remove:
        state_dict.pop(key, None)

    wavlm_model.load_state_dict(state_dict, strict=False)
    wavlm_model.eval()

    return wavlm_model, processor, calibrated_model, selector, device

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

def monte_carlo_dropout(model, input_values, num_samples=10):
    model.train()  
    mc_samples = []
    for _ in range(num_samples):
        with torch.no_grad():
            mc_samples.append(model(input_values).unsqueeze(0))
    mc_samples = torch.cat(mc_samples, dim=0)
    mean = torch.mean(mc_samples, dim=0)
    variance = torch.var(mc_samples, dim=0)
    return mean, variance

def predict_for_real(file_path, wavlm_model, processor, calibrated_model, selector, device):
    audio, sr = librosa.load(file_path, sr=16000, duration=60)
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        wavlm_features, wavlm_uncertainty = monte_carlo_dropout(wavlm_model, input_values)
    cqt_features, gfcc_features = extract_handcrafted_features(audio)
    
    wavlm_features_np = wavlm_features.cpu().numpy().flatten()
    cqt_features_np = cqt_features.flatten()
    gfcc_features_np = gfcc_features.flatten()
    wavlm_uncertainty_np = wavlm_uncertainty.cpu().numpy().flatten()

    # Normalize each feature set
    wavlm_norm = (wavlm_features_np - np.mean(wavlm_features_np)) / np.std(wavlm_features_np)
    cqt_norm = (cqt_features_np - np.mean(cqt_features_np)) / np.std(cqt_features_np)
    gfcc_norm = (gfcc_features_np - np.mean(gfcc_features_np)) / np.std(gfcc_features_np)

    wavlm_weight, cqt_weight, gfcc_weight = 0.4, 0.3, 0.3
    weighted_wavlm = wavlm_norm * wavlm_weight
    weighted_cqt = cqt_norm * cqt_weight
    weighted_gfcc = gfcc_norm * gfcc_weight
    combined_feature = np.concatenate([weighted_wavlm, weighted_cqt, weighted_gfcc, wavlm_uncertainty_np])
    selected_features = selector.transform(combined_feature.reshape(1, -1))
    
    spoof_probability = calibrated_model.predict_proba(selected_features)[0, 1]
    prediction = "Spoof" if spoof_probability > best_threshold else "Bonafide"
    
    return prediction, float(spoof_probability)
