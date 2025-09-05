from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from model import train_model, predict_attendance
from utils import combine_csv_files, preprocess_data
import joblib
import shutil
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})

# Global variables
combined_data = None
model_data = None
current_data_hash = None

def get_data_hash(data):
    """Generate a hash of the data to detect changes"""
    try:
        return str(pd.util.hash_pandas_object(data))
    except Exception as e:
        logger.error(f"Error generating data hash: {str(e)}")
        return str(hash(str(data.values.tobytes())))

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "success", "message": "Backend is running"}), 200

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    global combined_data, model_data, current_data_hash
    
    try:
        if 'files' not in request.files:
            return jsonify({"status": "error", "message": "No file part"}), 400
        
        files = request.files.getlist('files')
        if not files:
            return jsonify({"status": "error", "message": "No files selected"}), 400
        
        temp_dir = 'data'
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        file_paths = []
        
        try:
            for file in files:
                if file.filename.endswith('.csv'):
                    path = os.path.join(temp_dir, file.filename)
                    file.save(path)
                    file_paths.append(path)
            
            new_combined_data = combine_csv_files(file_paths)
            new_data_hash = get_data_hash(new_combined_data)
            
            if current_data_hash == new_data_hash:
                for path in file_paths:
                    os.remove(path)
                shutil.rmtree(temp_dir)
                
                return jsonify({
                    "status": "warning",
                    "message": "Same data as previous upload. No changes made.",
                    "data": {
                        "total_records": len(new_combined_data),
                        "columns": new_combined_data.columns.tolist()
                    }
                }), 200
            
            combined_data = new_combined_data
            current_data_hash = new_data_hash
            model_data = None
            
            for path in file_paths:
                os.remove(path)
            shutil.rmtree(temp_dir)
            
            return jsonify({
                "status": "success",
                "message": f"Successfully processed {len(file_paths)} CSV files",
                "data": {
                    "total_records": len(combined_data),
                    "columns": combined_data.columns.tolist(),
                    "data_hash": current_data_hash
                }
            }), 200
        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            logger.error(f"Error processing CSV files: {str(e)}")
            return jsonify({
                "status": "error",
                "message": "Failed to process CSV files",
                "details": str(e)
            }), 500
    except Exception as e:
        logger.error(f"Unexpected error in upload-csv: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Unexpected error during file upload",
            "details": str(e)
        }), 500

@app.route('/train', methods=['POST'])
def train():
    global combined_data, model_data
    
    try:
        if combined_data is None:
            return jsonify({
                "status": "error",
                "message": "No data available. Upload CSV first."
            }), 400
        
        logger.info("Starting model training process")
        logger.info(f"Data shape: {combined_data.shape}")
        logger.info(f"Data columns: {combined_data.columns.tolist()}")
        
        # Check if required columns exist
        required_columns = ['Attended_2021', 'Attended_2022', 'Attended_2023']
        missing_columns = [col for col in required_columns if col not in combined_data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return jsonify({
                "status": "error",
                "message": f"Missing required columns: {', '.join(missing_columns)}"
            }), 400
        
        try:
            processed_data, encoders = preprocess_data(combined_data)
            logger.info("Data preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Error during data preprocessing: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "status": "error",
                "message": "Failed to preprocess data",
                "details": str(e)
            }), 500
        
        try:
            model_data = train_model(processed_data, encoders)
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "status": "error",
                "message": "Failed to train model",
                "details": str(e)
            }), 500
        
        # Convert numpy values to Python types for JSON serialization
        model_data['data_hash'] = current_data_hash
        model_data['training_timestamp'] = pd.Timestamp.now().isoformat()
        model_data['data_shape'] = list(processed_data.shape)
        
        # Convert numpy values in model_metrics
        if 'model_metrics' in model_data:
            for key, value in model_data['model_metrics'].items():
                if isinstance(value, np.float64):
                    model_data['model_metrics'][key] = float(value)
                elif isinstance(value, np.int64):
                    model_data['model_metrics'][key] = int(value)
        
        # Convert optimal_threshold if it's a numpy type
        if 'optimal_threshold' in model_data and isinstance(model_data['optimal_threshold'], np.float64):
            model_data['optimal_threshold'] = float(model_data['optimal_threshold'])
        
        return jsonify({
            "status": "success",
            "message": "Model trained successfully",
            "data": {
                "model_metrics": model_data['model_metrics'],
                "optimal_threshold": model_data['optimal_threshold'],
                "data_hash": model_data['data_hash'],
                "training_timestamp": model_data['training_timestamp'],
                "data_shape": model_data['data_shape']
            }
        }), 200
    except Exception as e:
        logger.error(f"Unexpected error in train: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": "Failed to train model",
            "details": str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    global combined_data, model_data
    
    try:
        if model_data is None or combined_data is None:
            return jsonify({
                "status": "error",
                "message": "Model not trained or data not available"
            }), 400
        
        if model_data.get('data_hash') != current_data_hash:
            return jsonify({
                "status": "error",
                "message": "Model was trained on different data. Please retrain the model."
            }), 400
        
        data = request.get_json()
        conference_location = data.get('location')
        conference_specialization = data.get('specialization')
        
        if not conference_location or not conference_specialization:
            return jsonify({
                "status": "error",
                "message": "Location and specialization required"
            }), 400
        
        try:
            predictions = predict_attendance(
                combined_data, 
                model_data,
                conference_location,
                conference_specialization
            )
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "status": "error",
                "message": "Failed to make prediction",
                "details": str(e)
            }), 500
        
        predictions['prediction_metadata'] = {
            "data_hash": current_data_hash,
            "model_training_time": model_data.get('training_timestamp'),
            "prediction_time": pd.Timestamp.now().isoformat()
        }
        
        return jsonify({
            "status": "success",
            "message": "Prediction completed successfully",
            "data": predictions
        }), 200
    except Exception as e:
        logger.error(f"Unexpected error in predict: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": "Failed to make prediction",
            "details": str(e)
        }), 500

@app.route('/reset', methods=['POST'])
def reset():
    global combined_data, model_data, current_data_hash
    
    combined_data = None
    model_data = None
    current_data_hash = None
    
    if os.path.exists('data'):
        shutil.rmtree('data')
    
    model_files = ['optimized_rf_model.joblib', 'encoders.joblib']
    for file in model_files:
        if os.path.exists(file):
            os.remove(file)
    
    return jsonify({
        "status": "success",
        "message": "System reset successfully"
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)