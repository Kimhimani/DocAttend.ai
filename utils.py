import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import joblib

def combine_csv_files(file_paths):
    dataframes = []
    for path in file_paths:
        df = pd.read_csv(path)
        dataframes.append(df)
    
    combined = pd.concat(dataframes, ignore_index=True)
    return combined

def preprocess_data(data):
    # Handle missing values
    data.fillna({
        'Specialization': 'Unknown',
        'Location': 'Unknown',
        'Experience': data['Experience'].median() if 'Experience' in data.columns else 5
    }, inplace=True)
    
    # Ensure attendance columns exist, create them if they don't
    for year in ['2021', '2022', '2023']:
        col_name = f'Attended_{year}'
        if col_name not in data.columns:
            data[col_name] = 0  # Default to not attended
    
    # Encode categorical variables
    cat_cols = ['Specialization', 'Location']
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le
    
    # Feature engineering
    data['Attendance_Rate'] = (
        data['Attended_2021'] + 
        data['Attended_2022'] + 
        data['Attended_2023']
    ) / 3
    
    # Save encoders
    joblib.dump(encoders, 'encoders.joblib')
    
    return data, encoders