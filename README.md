# **For Real?: A Hybrid Approach to Audio Deepfake Detection in Noisy Real-World Environments**

***A joint project by: ADA, SI, CJP, ZSJ***

This repository contains the implementation of the thesis project titled ***"For Real?: A Hybrid Approach to Audio Deepfake Detection in Noisy Real-World Environments"***. The goal is to detect deepfake audio in noisy, real-world environments using a hybrid approach that combines handcrafted features with WavLM model features. The project includes a comparison between our proposed "For Real?" Model and a baseline model.

## Features

- Upload and analyze audio files
- Detect if audio is **Bonafide (real)** or **Spoofed (deepfake)**
- Uses a **Hybrid model** (WavLM + GFCC + CQT) with XGBoost classifier
- Option to switch to a **Handcrafted-only model** (GFCC + CQT) for comparison
- "Dark Mode" for preference
- "Simple Mode" for faster, lightweight use

## Datasets

### Training Data
- **ASVspoof 2019 LA (Train)**
### Evaluation Data
- **ASVspoof 2021 LA**  
- **ASVspoof 2021 DF**  
- **DeepVoice Dataset**  
- **For Real? Dataset** *(custom-recorded samples)*  
- **In-the-Wild Dataset** *(real-world noisy samples)* 

## Requirements
- Python 3.12+ (recommended)

## Installation
1. Clone the repository.
2. Set up the environment. (Optional but recommended to avoid package version conflicts)
3. Install the required Python packages by running: `pip install -r requirements.txt`
4. Navigate to ***models/for_real/*** and open ***WavLM Instruction.txt***. Follow the instructions to download ***best_wavlm_asvspoof_model.pth*** and place it in the same directory. This is necessary because GitHub does not allow uploading large files.
5. Download ***cards-loading-loop.webm*** at the google drive and place it in the ***static/images/***

## Running the Application
- Once everything is set up and no errors occur during the installation of dependencies, you can run the Flask app with the following command:
  `flask run` or `python app.py`. Visit the local web server URL that is shown in the terminal to access the website.
