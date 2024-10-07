from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
import time


# Import of the model functions
from src.for_real import predict_for_real, load_for_real_model
from src.baseline import predict_baseline, load_baseline_model

app = Flask(__name__)

# Nagchecheck if merong "uploads" folder
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Dito ilalagay ang mga uploaded audios at anong format ang tinatanggap
UPLOAD_FOLDER = 'uploads' 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'flac'}  

# Nagchecheck if accepted yung audio file format
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Dinedefine ang variable galing sa load model functions
wavlm_model, calibrated_model, selector, processor, device = load_for_real_model()
xgb_model = load_baseline_model()

# Global variable to store selected model; for_real as default
selected_model = 'for_real'

# Route papunta sa website natin (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# Route na naghahandle ng upload at prediction
@app.route('/predict', methods=['POST'])
def predict_audio():

    # Log time for monitoring
    start_time = time.time()

    if 'file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file securely
    filename = secure_filename(file.filename)
    audio_path = os.path.join("uploads", filename)
    file.save(audio_path)
    
    # Get the selected model type from the request headers (Default is For Real)
    selected_model = request.headers.get('Model-Type', 'proposed')

    # Debugging print to check current model
    print(f"Selected model for prediction: {selected_model}")

    try:
        # Select the appropriate model based on the user choice
        if selected_model == 'proposed':
            print("Running For Real model...")
            prediction, spoof_probability = predict_for_real(audio_path, wavlm_model, calibrated_model, selector, processor, device)
            print(f"Audio path: {audio_path}")
            
        elif selected_model == 'baseline':
            print("Running Baseline model...")
            prediction, spoof_probability = predict_baseline(audio_path, xgb_model)
            print(f"Audio path: {audio_path}")
            
        else:
            return jsonify({"error": "Invalid model selection"}), 400
        
        # Format the result to be returned
        result = {
            'model': selected_model,
            'audio': audio_path,
            'prediction': 'Spoof' if prediction == 1 else 'Bonafide',
            'probability': spoof_probability
        }

        # Log the time taken for the prediction
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken for prediction: {elapsed_time:.2f} seconds")

        # Delete the audio file after processing
        os.remove(audio_path)
        
        print("Prediction Result:", result)
        return jsonify(result)
    
    except Exception as e:
        print(f"Error encountered: {str(e)}")
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)