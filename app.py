from flask import Flask, request, render_template, jsonify, Response
import os
from werkzeug.utils import secure_filename
import time
import shutil
from datetime import datetime


# Import of the model functions
from src.for_real import evaluate_uploads, terminate_prediction, generate_progress, reset_progress, warmup_model, initialize_cuda
from src.baseline import load_baseline_model, predict_baseline_batch

app = Flask(__name__)

# Add model warmup during app initialization
print("Initializing CUDA and model...")
if initialize_cuda():
    warmup_success = warmup_model()
    if not warmup_success:
        print("Warning: Model warmup failed. First prediction may be slower.")
else:
    print("Warning: CUDA initialization failed. Performance may be affected.")

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
    start_time = time.time()

    if 'files[]' not in request.files:
        return jsonify({"error": "No audio files provided"}), 400
    
    files = request.files.getlist('files[]')
    if not files:
        return jsonify({"error": "No selected files"}), 400

    selected_model = request.headers.get('Model-Type', 'proposed')
    print(f"Selected model for prediction: {selected_model}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # Save all files first
        saved_paths = []
        original_names = {}  # Dictionary to store original filenames
        invalid_files = []
        
        for file in files:
            if file.filename == '':
                continue
                
            if not allowed_file(file.filename):
                invalid_files.append(file.filename)
                continue
            
            # Store timestamped filename and original filename mapping
            original_filename = secure_filename(file.filename)
            timestamped_filename = f"{timestamp}_{original_filename}"
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], timestamped_filename)
            
            file.save(audio_path)
            saved_paths.append(audio_path)
            original_names[audio_path] = original_filename  # Store mapping
        
        if not saved_paths:
            error_message = "No valid audio files found. Please upload WAV, MP3, or FLAC files only."
            return jsonify({"error": error_message}), 400

        if selected_model == 'proposed':
            print("Running For Real model...")
            gun = evaluate_uploads(app.config['UPLOAD_FOLDER'])
            if gun is None:
                for file in os.listdir(app.config['UPLOAD_FOLDER']):
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
                return jsonify({"error": "Prediction terminated"}), 499
            
            # Convert results using original filenames
            results = gun.magazine_to_json()
            converted_results = {}
            for path, result in results.items():
                original_name = original_names.get(path, os.path.basename(path))
                converted_results[original_name] = result
            results = converted_results
            
        elif selected_model == 'baseline':
            print("Running Baseline model...")
            results = predict_baseline_batch(saved_paths, xgb_model)
            # Convert baseline results too
            converted_results = {}
            for path, result in results.items():
                original_name = original_names.get(path, os.path.basename(path))
                converted_results[original_name] = result
            results = converted_results
        else:
            return jsonify({"error": "Invalid model selection"}), 400

        if invalid_files:
            invalid_files_message = f"Some files were skipped due to invalid format: {', '.join(invalid_files)}"
            results['message'] = invalid_files_message

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken for prediction: {elapsed_time:.2f} seconds")

        print("Processed Results:", results)
        
        # Clean up after successful processing
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
            
        return jsonify(results)
    
    except Exception as e:
        print(f"Error encountered: {str(e)}")
        # Clean up on error
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
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

@app.route('/progress')
def progress_stream():
    return Response(generate_progress(), mimetype='text/event-stream')

@app.route('/reset', methods=['POST'])
def reset():
    try:
        # Reset progress counters
        reset_progress()
        
        # Clean up uploads folder
        upload_folder = app.config['UPLOAD_FOLDER']
        for file in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        
        # Reset any other necessary state
        if 'eventSource' in globals():
            eventSource.close()
            
        return jsonify({"status": "Reset successful"}), 200
        
    except Exception as e:
        print(f"Error in reset: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=5000)