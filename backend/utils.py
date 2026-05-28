import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import joblib
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def combine_csv_files(file_paths):
    try:
        dataframes = []
        for path in file_paths:
            df = pd.read_csv(path)
            dataframes.append(df)
        
        combined = pd.concat(dataframes, ignore_index=True)
        return combined
    except Exception as e:
        logger.error(f"Error combining CSV files: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def preprocess_data(data):
    try:
        logger.info("Starting data preprocessing")
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Input data columns: {data.columns.tolist()}")
        
        # Handle missing values
        data.fillna({
            'Specialization': 'Unknown',
            'Location': 'Unknown',
            'Experience': data['Experience'].median() if 'Experience' in data.columns else 5
        }, inplace=True)
        
        # Ensure attendance columns exist, create them if they don't
        attendance_columns = ['Attended_2021', 'Attended_2022', 'Attended_2023']
        for col in attendance_columns:
            if col not in data.columns:
                data[col] = 0  # Default to not attended
                logger.warning(f"Column {col} not found, created with default value 0")
        
        # Convert text-based attendance to binary
        for col in attendance_columns:
            # Convert to lowercase for case-insensitive matching
            data[col] = data[col].astype(str).str.lower()
            
            # Map common text representations to binary
            attendance_map = {
                'yes': 1, 'present': 1, 'attended': 1, 'true': 1, '1': 1,
                'no': 0, 'absent': 0, 'not attended': 0, 'false': 0, '0': 0
            }
            
            # Apply mapping, default to 0 for unknown values
            data[col] = data[col].map(attendance_map).fillna(0).astype(int)
        
        # Encode categorical variables
        cat_cols = ['Specialization', 'Location']
        encoders = {}
        
        for col in cat_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            encoders[col] = le
            logger.info(f"Encoded {col}: {len(le.classes_)} classes")
        
        # Feature engineering
        data['Attendance_Rate'] = (
            data['Attended_2021'] + 
            data['Attended_2022'] + 
            data['Attended_2023']
        ) / 3
        
        # Save encoders
        joblib.dump(encoders, 'encoders.joblib')
        
        logger.info("Data preprocessing completed")
        return data, encoders
    except Exception as e:
        logger.error(f"Error during data preprocessing: {str(e)}")
        logger.error(traceback.format_exc())
        raise