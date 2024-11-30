from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
import time
import shutil


# Import of the model functions
from src.for_real import evaluate_uploads, terminate_prediction
from src.baseline import load_baseline_model, predict_baseline_batch

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
xgb_model = load_baseline_model()

# Global variable to store selected model; for_real as default
selected_model = 'for_real'

# Route papunta sa website natin (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# Route papunta sa simple.html
@app.route('/simple')
def simple():
    return render_template('simple.html')

# Route na naghahandle ng upload at prediction
@app.route('/predict', methods=['POST'])
def predict_audio():

    # Log time for monitoring
    start_time = time.time()

    # Clear the uploads folder before processing new files
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))

    if 'files[]' not in request.files:
        return jsonify({"error": "No audio files provided"}), 400
    
    files = request.files.getlist('files[]')
    if not files:
        return jsonify({"error": "No selected files"}), 400

    # Get the selected model type
    selected_model = request.headers.get('Model-Type', 'proposed')
    print(f"Selected model for prediction: {selected_model}")

    try:
        # Save all files first
        saved_paths = []
        invalid_files = []
        
        for file in files:
            if file.filename == '':
                continue
                
            if not allowed_file(file.filename):
                invalid_files.append(file.filename)
                continue
            
            filename = secure_filename(file.filename)
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(audio_path)
            saved_paths.append(audio_path)
        
        # Check if we have any valid files to process
        if not saved_paths:
            error_message = "No valid audio files found. Please upload WAV, MP3, or FLAC files only."
            return jsonify({"error": error_message}), 400

        # Process valid files based on selected model
        if selected_model == 'proposed':
            print("Running For Real model...")
            gun = evaluate_uploads(app.config['UPLOAD_FOLDER'])
            if gun is None:
                return jsonify({"error": "Prediction terminated"}), 499
            results = gun.magazine_to_json()
            
        elif selected_model == 'baseline':
            print("Running Baseline model...")
            results = predict_baseline_batch(saved_paths, xgb_model)
            
        else:
            return jsonify({"error": "Invalid model selection"}), 400

        # Add information about invalid files to results if any
        if invalid_files:
            invalid_files_message = f"Some files were skipped due to invalid format: {', '.join(invalid_files)}"
            results['message'] = invalid_files_message

        # Log processing time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken for prediction: {elapsed_time:.2f} seconds")

        # Clean up the uploads folder
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))

        print("Processed Results:", results)
        return jsonify(results)
    
    except Exception as e:
        print(f"Error encountered: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/terminate', methods=['POST'])
def terminate():
    try:
        # First terminate the prediction
        terminate_prediction()
        
        # Clean up uploads folder
        upload_folder = app.config['UPLOAD_FOLDER']
        for file in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        
        return jsonify({
            "status": "Prediction terminated and files cleaned up"
        }), 200
        
    except Exception as e:
        print(f"Error in terminate: {e}")
        return jsonify({
            "error": f"Error during termination: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host="0.0.0.0", port=5000)