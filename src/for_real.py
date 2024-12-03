import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import io
from pathlib import Path
import librosa
import torch.nn.functional as F
from spafe.features.gfcc import gfcc
import joblib
from transformers import AutoProcessor, WavLMModel
from datetime import datetime
import signal
from functools import wraps
import threading
from flask import Response
import json
import time
from flask import Flask, Response  # Make sure Flask is imported

# Global flag for termination
TERMINATE_PREDICTION = threading.Event()

# Global variables for progress tracking
current_progress = 0
total_files = 0
processed_files = 0

# Add global variables for model caching
_model_cache = {
    'feature_extractor': None,
    'model_bundle': None
}

def terminate_prediction():
    global current_progress, total_files, processed_files
    
    TERMINATE_PREDICTION.set()
    
    # Reset progress tracking
    current_progress = 0
    total_files = 0
    processed_files = 0
    
    print("Terminating prediction...")

def generate_progress():
    global current_progress, total_files, processed_files
    
    # Reset at start of progress tracking
    current_progress = 0
    
    while True:
        if total_files > 0:
            # Calculate progress based on completed files and cap at 100%
            current_progress = min(int((processed_files / total_files) * 100), 100)
            
            data = json.dumps({
                "progress": current_progress,
                "processed": min(processed_files, total_files),
                "total": total_files
            })
            yield f"data: {data}\n\n"
            
            if current_progress >= 100:
                # Reset all tracking variables
                current_progress = 0
                total_files = 0
                processed_files = 0
                break
        else:
            # If no files are being processed, send zero progress
            data = json.dumps({
                "progress": 0,
                "processed": 0,
                "total": 0
            })
            yield f"data: {data}\n\n"
        
        time.sleep(0.2)

# Decorator to check for termination
def check_termination(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if TERMINATE_PREDICTION.is_set():
            print(f"Terminating {func.__name__}")
            return None
        return func(*args, **kwargs)
    return wrapper

# Basic setup
device = torch.device("cpu")
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

model_name = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
processor = AutoProcessor.from_pretrained(model_name)

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
    def __init__(self, pretrained_weights_path):
        super(CustomWavLMForFeatureExtraction, self).__init__()
        
        self.wavlm = WavLMModel.from_pretrained(model_name)
        used_keys = ['feature_extractor', 'feature_projection', 'encoder']
        filtered_state_dict = {k: v for k, v in self.wavlm.state_dict().items() 
        if any(uk in k for uk in used_keys)}
        self.wavlm.load_state_dict(filtered_state_dict, strict=False)
        self.attention = AttentionLayer(self.wavlm.config.hidden_size)
        
        pretrained_dict = torch.load(pretrained_weights_path, map_location=device)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def enable_dropout(self):
        for m in self.wavlm.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def monte_carlo_forward(self, input_values, n_samples=5):
        self.eval()
        self.enable_dropout()
        
        # Store original batch size and input shape
        batch_size = input_values.size(0)
        
        # Create list to store outputs
        mc_outputs = []
        
        with torch.no_grad():
            # Run n forward passes
            for _ in range(n_samples):
                outputs = self.wavlm(input_values)
                hidden_states = outputs.last_hidden_state
                attended_hidden = self.attention(hidden_states)
                mc_outputs.append(attended_hidden)
            
            # Stack the outputs [n_samples, batch_size, hidden_size]
            mc_outputs = torch.stack(mc_outputs)
        
        mean = torch.mean(mc_outputs, dim=0)
        variance = torch.var(mc_outputs, dim=0)
        return mean, variance

    def forward(self, input_values, attention_mask=None):
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        attended_hidden = self.attention(hidden_states)
        return attended_hidden

class Bullet:
    def __init__(self, audio_path, prediction, probability, threshold, processing_time):
        self.audio_path = Path(audio_path)
        self.prediction = prediction
        self.probability = probability
        self.threshold = threshold
        self.processing_time = processing_time

    def __str__(self):
        return (f"Bullet loaded with {self.audio_path.name}:\n"
                f"• Prediction: {self.prediction}\n"
                f"• Probability: {self.probability:.4f}\n"
                f"• Threshold: {self.threshold:.4f}\n"
                f"• Processing Time: {self.processing_time:.2f}s")

class Gun:
    def __init__(self):
        self.chambers = 0  # Dynamic number of chambers (up to 10)
        self.fired_chambers = 0
        self.magazine = {}  # Store bullets with their results
        
    def load_magazine(self, num_bullets):
        """Initialize revolver with number of chambers needed"""
        self.chambers = min(num_bullets, 10)  # Maximum 10 chambers
        self.fired_chambers = 0
        print(f"LOADING {self.chambers}-chamber revolver!")
    
    def fire_round(self, audio_path=None, result=None):
        """Fire a round and store the bullet's result"""
        self.fired_chambers += 1
        
        print(f"\nChamber {self.fired_chambers}/{self.chambers} fired")
        
        if audio_path and result:
            bullet = Bullet(
                audio_path,
                result['prediction'],
                result['probability'],
                result['threshold'],
                result['processing_time']
            )
            self.magazine[audio_path] = bullet
            print(f"{bullet}")  # Display bullet result immediately after firing
        
        if self.check_trigger():
            self.pull_trigger()
    
    def check_trigger(self):
        """Check if all chambers have been fired"""
        return self.fired_chambers == self.chambers
    
    def pull_trigger(self):
        """Trigger completion handler"""
        print(f"BANG! All {self.chambers} chambers fired!")
    
    def inspect_magazine(self):
        """Get all bullets"""
        return self.magazine
    
    def inspect_bullet(self, audio_path):
        """Get specific bullet"""
        return self.magazine.get(audio_path)
    
    def magazine_to_json(self):
        # Create a dictionary where each bullet's data is converted to JSON-compatible format
        return {
            str(audio_path): {
                "audio_path": str(bullet.audio_path),
                "prediction": bullet.prediction,
                "probability": bullet.probability,
                "threshold": bullet.threshold,
                "processing_time": bullet.processing_time
            }
            for audio_path, bullet in self.magazine.items()
        }

def warmup_model():
    """Perform model warmup with dummy data"""
    print("Warming up model...")
    try:
        # Initialize model if not already cached
        if not _model_cache['feature_extractor']:
            model_bundle_path = os.path.join("models", "for_real", "calibrated_model_20241203_141204.joblib")
            pretrained_weights_path = os.path.join("models", "for_real", "best_wavlm_asvspoof_model.pth")
            
            # Cache the models
            _model_cache['model_bundle'] = joblib.load(model_bundle_path)
            _model_cache['feature_extractor'] = CustomWavLMForFeatureExtraction(pretrained_weights_path).to(device)
            _model_cache['feature_extractor'].eval()

        # Create multiple dummy inputs of different lengths for better warmup
        dummy_lengths = [16000 * length for length in [1, 5, 10, 30, 60]]  # 1s, 5s, 10s, 30s, 60s
        
        with torch.cuda.amp.autocast():
            for length in dummy_lengths:
                # Create dummy input with correct shape [batch_size, sequence_length]
                dummy_input = torch.randn(1, length).to(device)
                
                # Perform dummy forward pass with Monte Carlo
                _ = _model_cache['feature_extractor'].monte_carlo_forward(dummy_input)
                
                # Clear CUDA cache after each forward pass
                torch.cuda.empty_cache()
            
        print("Model warmup complete!")
        return True
        
    except Exception as e:
        print(f"Error during model warmup: {e}")
        return False

# Add this to ensure CUDA is initialized
def initialize_cuda():
    """Initialize CUDA to prevent first-time overhead"""
    try:
        # Force CUDA initialization
        torch.cuda.init()
        # Perform a small dummy operation
        dummy = torch.zeros(1).cuda()
        del dummy
        torch.cuda.empty_cache()
        print("CUDA initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing CUDA: {e}")
        return False

class SingleSampleEvaluator:
    def __init__(self, model_bundle_path, pretrained_weights_path, target_sr=16000, threshold_offset=0.0):
        self.target_sr = target_sr
        
        # Use cached models if available
        if _model_cache['feature_extractor'] is None:
            self.feature_extractor = self._initialize_feature_extractor(pretrained_weights_path)
            _model_cache['feature_extractor'] = self.feature_extractor
        else:
            self.feature_extractor = _model_cache['feature_extractor']
            
        if _model_cache['model_bundle'] is None:
            self.model_bundle = joblib.load(model_bundle_path)
            _model_cache['model_bundle'] = self.model_bundle
        else:
            self.model_bundle = _model_cache['model_bundle']
            
        self.model = self.model_bundle['model']
        self.threshold = self.model_bundle['threshold'] + threshold_offset
        self.gun = Gun()
        self._feature_cache = {}
        print(f"Loaded model bundle with threshold: {self.threshold:.4f} (base: {self.model_bundle['threshold']:.4f}, offset: {threshold_offset:.4f})")

    def _initialize_feature_extractor(self, pretrained_weights_path):
        feature_extractor = CustomWavLMForFeatureExtraction(pretrained_weights_path).to(device)
        feature_extractor.eval()
        return feature_extractor

    @check_termination
    def process_audio(self, audio_path):
        if TERMINATE_PREDICTION.is_set():
            return None
        try:
            # Load audio
            with open(audio_path, 'rb') as f:
                waveform, sr = torchaudio.load(io.BytesIO(f.read()))
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Calculate duration in seconds
            duration = waveform.shape[1] / self.target_sr
            
            # Calculate number of full 5-second segments needed
            segment_length = 5 * self.target_sr  # 5 seconds in samples
            segments_needed = 12  # 60 seconds / 5 seconds = 12 segments
            
            # Split audio into 5-second segments
            valid_segments = []
            num_full_segments = waveform.shape[1] // segment_length
            
            for i in range(num_full_segments):
                start = i * segment_length
                end = start + segment_length
                segment = waveform[:, start:end]
                if segment.shape[1] == segment_length:  # Only keep full 5-second segments
                    valid_segments.append(segment)
            
            # If we have enough segments, randomly select what we need
            if len(valid_segments) >= segments_needed:
                # Randomly select segments without replacement
                selected_indices = torch.randperm(len(valid_segments))[:segments_needed]
                selected_segments = [valid_segments[i] for i in selected_indices]
            
                # Concatenate selected segments
                processed_waveform = torch.cat(selected_segments, dim=1)
            
            else:
                # If we don't have enough segments, use what we have and pad
                if valid_segments:
                    processed_waveform = torch.cat(valid_segments, dim=1)
                else:
                    processed_waveform = waveform
                
                # Pad or trim to target length (60 seconds)
                target_length = self.target_sr * 60
                if processed_waveform.shape[1] > target_length:
                    processed_waveform = processed_waveform[:, :target_length]
                else:
                    processed_waveform = F.pad(processed_waveform, (0, target_length - processed_waveform.shape[1]))
            
            # Extract features
            features = self._extract_features(processed_waveform)
            
            return self._combine_features(features)
            
        except Exception as e:
            print(f"Error processing audio file: {str(e)}")
            return None

    def _extract_features(self, waveform):
        waveform_gpu = waveform.to(device)
        
        with torch.cuda.amp.autocast():
            # WavLM features
            wavlm_mean, wavlm_var = self.feature_extractor.monte_carlo_forward(waveform_gpu)
            
            # Handcrafted features on CPU
            cpu_waveform = waveform.cpu()
            cqt_features = self.extract_cqt(cpu_waveform)
            
            gfcc_features = self.extract_gfcc(cpu_waveform)
        
        return {
            'wavlm_mean': wavlm_mean.cpu(),
            'wavlm_var': wavlm_var.cpu(),
            'cqt': cqt_features,
            'gfcc': gfcc_features
        }

    def _combine_features(self, features):
        combined_features = torch.cat([
            features['wavlm_mean'].flatten(),    # 768 features
            features['wavlm_var'].flatten(),     # 768 features
            features['cqt'].flatten(),           # 336 features
            features['gfcc'].flatten()           # 80 features
        ], dim=0)
        
        return combined_features.numpy().reshape(1, -1)

    def extract_cqt(self, waveform):
        """Extract CQT features with the same dimensionality as training"""
        audio_np = waveform.squeeze().numpy()
        cqt = librosa.cqt(
            audio_np, 
            sr=self.target_sr,
            hop_length=512,
            n_bins=84,
            bins_per_octave=12
        )
        # Extract statistics across time dimension to match training
        cqt_features = np.hstack([
            np.mean(np.abs(cqt), axis=1),  # 84 features
            np.std(np.abs(cqt), axis=1),   # 84 features
            np.max(np.abs(cqt), axis=1),   # 84 features
            np.min(np.abs(cqt), axis=1)    # 84 features
        ])
        return torch.from_numpy(cqt_features)  # Total: 336 features

    def extract_gfcc(self, waveform, num_ceps=20):
        """Extract GFCC features with the same dimensionality as training"""
        audio_np = waveform.squeeze().numpy()
        gfcc_features = gfcc(
            sig=audio_np,
            fs=self.target_sr,
            num_ceps=num_ceps,  # Changed from 80 to 20
            nfilts=40,          # Changed from 128 to 40
            nfft=2048,
            low_freq=0,
            high_freq=None,
            dct_type=2,
            use_energy=False,
            lifter=None,
            normalize=None
        )
        # Extract statistics across time dimension to match training
        gfcc_stats = np.hstack([
            np.mean(gfcc_features, axis=0),  # 20 features
            np.std(gfcc_features, axis=0),   # 20 features
            np.max(gfcc_features, axis=0),   # 20 features
            np.min(gfcc_features, axis=0)    # 20 features
        ])
        return torch.from_numpy(gfcc_stats)  # Total: 80 features

    @check_termination
    def evaluate_batch(self, audio_paths):
        global processed_files, total_files
        
        if TERMINATE_PREDICTION.is_set():
            processed_files = 0
            total_files = 0
            return None
        
        # Don't set processed_files to 0 here since it's handled in evaluate_folder
        self.gun.load_magazine(len(audio_paths))
        results = {}

        try:
            for audio_path in audio_paths:
                if processed_files >= total_files:
                    break
                
                result = self.evaluate(audio_path)
                if result:
                    results[audio_path] = result
                    self.gun.fire_round(audio_path, result)
                    # Only increment after successful processing AND firing
                    processed_files = min(processed_files + 1, total_files)
                else:
                    self.gun.fire_round()
                
                print(f"Progress: {processed_files}/{total_files}")
            
            return results
        except Exception as e:
            processed_files = 0
            total_files = 0
            print(f"Error in evaluate_batch: {str(e)}")
            return None

    def evaluate_folder(self, folder_path, supported_extensions=('.mp3', '.wav', '.flac')):
        """Process all supported audio files in a folder"""
        global total_files, processed_files, current_progress
        
        print("\n=== Starting evaluate_folder ===")
        print(f"Folder path: {os.path.abspath(folder_path)}")
        
        # Reset all counters at the start of each folder evaluation
        total_files = 0
        processed_files = 0
        current_progress = 0
        
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"Error: Folder not found: {folder_path}")
            return {}
        
        # Get all audio files in folder
        audio_files = []
        print("\nScanning for audio files:")
        for ext in supported_extensions:
            found_files = list(folder_path.glob(f"*{ext}"))
            print(f"- Found {len(found_files)} {ext} files")
            audio_files.extend(found_files)
        
        if not audio_files:
            print(f"No supported audio files found in {folder_path}")
            print(f"Supported formats: {', '.join(supported_extensions)}")
            print("\nListing all files in directory:")
            try:
                all_files = list(folder_path.glob("*"))
                for file in all_files:
                    print(f"- {file.name} ({file.suffix})")
            except Exception as e:
                print(f"Error listing directory: {e}")
            return {}
        
        # Limit to 10 files if necessary
        if len(audio_files) > 10:
            print(f"Warning: Found {len(audio_files)} files. Processing first 10 only.")
            audio_files = audio_files[:10]
        
        # Set total_files here
        total_files = len(audio_files)
        
        print(f"\nProcessing {total_files} files from: {folder_path}")
        print(f"Supported formats: {', '.join(supported_extensions)}")
        
        return self.evaluate_batch(audio_files)

    @check_termination
    def evaluate(self, audio_path):
        if TERMINATE_PREDICTION.is_set():
            return None
        
        try:
            # Record start time
            start_time = datetime.now()
            
            # Process the audio file
            result = self.process_audio(audio_path)
            if result is None:
                return None
            
            # Get probability and make prediction
            probability = self.model.predict_proba(result)[0, 1]
            prediction = int(probability >= self.threshold)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'prediction': 'spoof' if prediction == 1 else 'bonafide',
                'probability': float(probability),
                'threshold': float(self.threshold),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'processing_time': processing_time
            }
            
        except Exception as e:
            print(f"Error processing audio file: {str(e)}")
            return None

def evaluate_uploads(upload_folder):
    global current_progress, total_files, processed_files
    
    print("\n=== Starting evaluate_uploads ===")
    print(f"Upload folder path: {os.path.abspath(upload_folder)}")
    
    # Debug: Print folder contents
    try:
        if not os.path.exists(upload_folder):
            print(f"Upload folder does not exist: {upload_folder}")
            os.makedirs(upload_folder)
            print(f"Created upload folder: {upload_folder}")
        
        files = os.listdir(upload_folder)
        print(f"Found {len(files)} files:")
        for file in files:
            file_path = os.path.join(upload_folder, file)
            print(f"- {file} ({os.path.getsize(file_path)} bytes)")
    except Exception as e:
        print(f"Error checking upload folder: {e}")
    
    # Reset all progress tracking variables at start
    current_progress = 0
    total_files = 0
    processed_files = 0
    
    # Reset termination flag at start
    TERMINATE_PREDICTION.clear()
    
    try:
        # Configuration - Update paths to use calibrated model
        model_bundle_path = os.path.join("models", "for_real", "calibrated_model_20241203_141204.joblib")
        pretrained_weights_path = os.path.join("models", "for_real", "best_wavlm_asvspoof_model.pth")

        # Load model bundle once
        model_bundle = joblib.load(model_bundle_path)
        base_threshold = model_bundle['threshold']  # This will now be 0.475 from calibrated model
        print(f"\nUsing calibrated model threshold: {base_threshold:.4f}")

        # Store in cache to avoid reloading
        _model_cache['model_bundle'] = model_bundle

        evaluator = SingleSampleEvaluator(
            model_bundle_path, 
            pretrained_weights_path,
            threshold_offset=0.0053 # No need for offset since we're using calibrated threshold
        )
        
        # Process files and store results
        results = evaluator.evaluate_folder(upload_folder)
        
        # Only return the gun if we actually processed some files
        if results:
            return evaluator.gun
        else:
            print("No results were generated")
            return None
        
    except Exception as e:
        print(f"Error in evaluate_uploads: {str(e)}")
        return None

def reset_progress():
    global current_progress, total_files, processed_files
    
    # Reset counters
    current_progress = 0
    total_files = 0
    processed_files = 0
