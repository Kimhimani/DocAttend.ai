from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from model import train_model, predict_attendance
from utils import combine_csv_files, preprocess_data

app = Flask(__name__)
# Configure CORS to allow requests from the frontend
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})

# Global variables to store data and model
combined_data = None
model = None
encoders = None
feature_names = None

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify backend is accessible"""
    return jsonify({"status": "success", "message": "Backend is running"}), 200

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    global combined_data
    if 'files' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file part"
        }), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({
            "status": "error",
            "message": "No files selected"
        }), 400
    
    # Save uploaded files temporarily
    temp_dir = 'data'
    os.makedirs(temp_dir, exist_ok=True)
    file_paths = []
    
    try:
        for file in files:
            if file.filename.endswith('.csv'):
                path = os.path.join(temp_dir, file.filename)
                file.save(path)
                file_paths.append(path)
        
        # Combine CSV files
        combined_data = combine_csv_files(file_paths)
        
        # Clean up temporary files
        for path in file_paths:
            os.remove(path)
        
        return jsonify({
            "status": "success",
            "message": f"Successfully combined {len(file_paths)} CSV files",
            "data": {
                "total_records": len(combined_data),
                "columns": combined_data.columns.tolist()
            }
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to process CSV files",
            "details": str(e)
        }), 500

@app.route('/train', methods=['POST'])
def train():
    global combined_data, model, encoders, feature_names
    
    try:
        if combined_data is None:
            return jsonify({
                "status": "error",
                "message": "No data available. Upload CSV first."
            }), 400
        
        # Preprocess data
        processed_data, encoders = preprocess_data(combined_data)
        
        # Train model
        model, metrics, feature_names = train_model(processed_data)
        
        return jsonify({
            "status": "success",
            "message": "Model trained successfully",
            "data": {
                "metrics": metrics
            }
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to train model",
            "details": str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    global combined_data, model, encoders, feature_names
    
    try:
        if model is None or combined_data is None:
            return jsonify({
                "status": "error",
                "message": "Model not trained or data not available"
            }), 400
        
        data = request.get_json()
        conference_location = data.get('location')
        conference_specialization = data.get('specialization')
        
        if not conference_location or not conference_specialization:
            return jsonify({
                "status": "error",
                "message": "Location and specialization required"
            }), 400
        
        # Make predictions
        predictions = predict_attendance(
            combined_data, 
            model, 
            encoders,
            feature_names,
            conference_location,
            conference_specialization
        )
        
        return jsonify({
            "status": "success",
            "message": "Prediction completed successfully",
            "data": predictions
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to make prediction",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)