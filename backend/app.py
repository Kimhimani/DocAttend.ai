from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from model import train_model, predict_attendance
from utils import combine_csv_files, preprocess_data
import shutil
import logging
import traceback

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# ----------------------------
# Flask App Initialization
# ----------------------------
app = Flask(__name__)

# Enable CORS for all origins
CORS(app)

# ----------------------------
# Global Variables
# ----------------------------
combined_data = None
model_data = None
current_data_hash = None

UPLOAD_FOLDER = "data"

# ----------------------------
# Utility Functions
# ----------------------------
def get_data_hash(data):
    """
    Generate a hash of dataframe to detect duplicate uploads
    """
    try:
        return str(pd.util.hash_pandas_object(data).sum())
    except Exception as e:
        logger.error(f"Hash generation error: {str(e)}")
        return str(hash(str(data.values.tobytes())))


def cleanup_temp_folder():
    """
    Remove temporary upload folder safely
    """
    try:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")


# ----------------------------
# Health Check Route
# ----------------------------
@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "DocAttend Backend Running Successfully"
    })


@app.route("/test", methods=["GET"])
def test():
    return jsonify({
        "status": "success",
        "message": "Backend is running"
    })


# ----------------------------
# Upload CSV Route
# ----------------------------
@app.route("/upload-csv", methods=["POST"])
def upload_csv():
    global combined_data, model_data, current_data_hash

    try:
        if "files" not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file part found"
            }), 400

        files = request.files.getlist("files")

        if len(files) == 0:
            return jsonify({
                "status": "error",
                "message": "No files selected"
            }), 400

        # Create temp upload folder
        cleanup_temp_folder()
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        file_paths = []

        # Save uploaded files
        for file in files:

            if file.filename == "":
                continue

            if not file.filename.endswith(".csv"):
                continue

            file_path = os.path.join(UPLOAD_FOLDER, file.filename)

            file.save(file_path)

            file_paths.append(file_path)

        if len(file_paths) == 0:
            return jsonify({
                "status": "error",
                "message": "No valid CSV files uploaded"
            }), 400

        # Combine CSVs
        new_combined_data = combine_csv_files(file_paths)

        # Generate hash
        new_data_hash = get_data_hash(new_combined_data)

        # Detect duplicate uploads
        if current_data_hash == new_data_hash:

            cleanup_temp_folder()

            return jsonify({
                "status": "warning",
                "message": "Same dataset already uploaded",
                "data": {
                    "total_records": len(new_combined_data),
                    "columns": new_combined_data.columns.tolist()
                }
            }), 200

        # Update global variables
        combined_data = new_combined_data
        current_data_hash = new_data_hash
        model_data = None

        cleanup_temp_folder()

        return jsonify({
            "status": "success",
            "message": f"{len(file_paths)} CSV files processed successfully",
            "data": {
                "total_records": len(combined_data),
                "columns": combined_data.columns.tolist(),
                "data_hash": current_data_hash
            }
        }), 200

    except Exception as e:

        cleanup_temp_folder()

        logger.error(traceback.format_exc())

        return jsonify({
            "status": "error",
            "message": "Failed to process CSV files",
            "details": str(e)
        }), 500


# ----------------------------
# Train Model Route
# ----------------------------
@app.route("/train", methods=["POST"])
def train():
    global combined_data, model_data

    try:

        if combined_data is None:
            return jsonify({
                "status": "error",
                "message": "Upload CSV files first"
            }), 400

        logger.info("Training model started")

        required_columns = [
            "Attended_2021",
            "Attended_2022",
            "Attended_2023"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in combined_data.columns
        ]

        if missing_columns:
            return jsonify({
                "status": "error",
                "message": f"Missing columns: {missing_columns}"
            }), 400

        # Preprocess data
        processed_data, encoders = preprocess_data(combined_data)

        logger.info("Preprocessing completed")

        # Train model
        model_data = train_model(processed_data, encoders)

        logger.info("Model training completed")

        # Add metadata
        model_data["data_hash"] = current_data_hash
        model_data["training_timestamp"] = pd.Timestamp.now().isoformat()
        model_data["data_shape"] = list(processed_data.shape)

        # Convert numpy types
        if "model_metrics" in model_data:

            cleaned_metrics = {}

            for key, value in model_data["model_metrics"].items():

                if isinstance(value, np.integer):
                    cleaned_metrics[key] = int(value)

                elif isinstance(value, np.floating):
                    cleaned_metrics[key] = float(value)

                else:
                    cleaned_metrics[key] = value

            model_data["model_metrics"] = cleaned_metrics

        if isinstance(model_data.get("optimal_threshold"), np.floating):
            model_data["optimal_threshold"] = float(
                model_data["optimal_threshold"]
            )

        return jsonify({
            "status": "success",
            "message": "Model trained successfully",
            "data": {
                "model_metrics": model_data["model_metrics"],
                "optimal_threshold": model_data["optimal_threshold"],
                "training_timestamp": model_data["training_timestamp"],
                "data_shape": model_data["data_shape"]
            }
        }), 200

    except Exception as e:

        logger.error(traceback.format_exc())

        return jsonify({
            "status": "error",
            "message": "Training failed",
            "details": str(e)
        }), 500


# ----------------------------
# Prediction Route
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    global combined_data, model_data

    try:

        if combined_data is None or model_data is None:
            return jsonify({
                "status": "error",
                "message": "Train model before prediction"
            }), 400

        if model_data.get("data_hash") != current_data_hash:
            return jsonify({
                "status": "error",
                "message": "Dataset changed. Retrain model."
            }), 400

        data = request.get_json()

        conference_location = data.get("location")
        conference_specialization = data.get("specialization")

        if not conference_location or not conference_specialization:
            return jsonify({
                "status": "error",
                "message": "Location and specialization required"
            }), 400

        predictions = predict_attendance(
            combined_data,
            model_data,
            conference_location,
            conference_specialization
        )

        predictions["prediction_metadata"] = {
            "prediction_time": pd.Timestamp.now().isoformat(),
            "training_time": model_data.get("training_timestamp")
        }

        return jsonify({
            "status": "success",
            "message": "Prediction completed successfully",
            "data": predictions
        }), 200

    except Exception as e:

        logger.error(traceback.format_exc())

        return jsonify({
            "status": "error",
            "message": "Prediction failed",
            "details": str(e)
        }), 500


# ----------------------------
# Reset Route
# ----------------------------
@app.route("/reset", methods=["POST"])
def reset():
    global combined_data, model_data, current_data_hash

    try:

        combined_data = None
        model_data = None
        current_data_hash = None

        cleanup_temp_folder()

        model_files = [
            "optimized_rf_model.joblib",
            "encoders.joblib"
        ]

        for file in model_files:

            if os.path.exists(file):
                os.remove(file)

        return jsonify({
            "status": "success",
            "message": "System reset successful"
        })

    except Exception as e:

        logger.error(traceback.format_exc())

        return jsonify({
            "status": "error",
            "message": "Reset failed",
            "details": str(e)
        }), 500


# ----------------------------
# Main Entry
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)