import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

def train_model(processed_data):
    # Define features and target
    X = processed_data.drop(['Attended_2023', 'Doctor_ID', 'Name', 'Email', 'Phone'], axis=1)
    y = processed_data['Attended_2023']
    
    # Store feature names for later use during prediction
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Save model and feature names together
    joblib.dump((model, feature_names), 'model.joblib')
    
    return model, metrics, feature_names

def predict_attendance(data, model, encoders, feature_names, conf_location, conf_specialization):
    # Create a copy of the data for prediction
    prediction_data = data.copy()
    
    # Encode conference location and specialization
    try:
        conf_location_encoded = encoders['Location'].transform([conf_location])[0]
    except ValueError:
        # If location not seen before, use the most common one
        conf_location_encoded = data['Location'].mode()[0]
    
    try:
        conf_specialization_encoded = encoders['Specialization'].transform([conf_specialization])[0]
    except ValueError:
        # If specialization not seen before, use the most common one
        conf_specialization_encoded = data['Specialization'].mode()[0]
    
    # Create features for prediction (matching training features)
    features = pd.DataFrame()
    features['Specialization'] = prediction_data['Specialization']
    features['Location'] = prediction_data['Location']
    features['Experience'] = prediction_data['Experience']
    
    # Calculate attendance rate only if the columns exist
    attendance_columns = [col for col in ['Attended_2021', 'Attended_2022', 'Attended_2023'] if col in prediction_data.columns]
    
    if attendance_columns:
        features['Attendance_Rate'] = prediction_data[attendance_columns].sum(axis=1) / len(attendance_columns)
    else:
        # If no attendance columns exist, use a default value
        features['Attendance_Rate'] = 0.5  # Neutral value
    
    # Ensure features are in the same order as during training
    available_features = [col for col in feature_names if col in features.columns]
    missing_features = [col for col in feature_names if col not in features.columns]
    
    # Add any missing features with default values
    for col in missing_features:
        features[col] = 0  # Default value
    
    # Reorder columns to match training
    features = features[feature_names]
    
    # Make predictions (base probability)
    base_probabilities = model.predict_proba(features)[:, 1]
    
    # Adjust probabilities based on conference match
    adjusted_probabilities = base_probabilities.copy()
    
    # Boost factors for location and specialization match
    location_boost = 0.3  # 30% increase if location matches
    specialization_boost = 0.4  # 40% increase if specialization matches
    
    for i, row in prediction_data.iterrows():
        # Check if doctor's location matches conference location
        if row['Location'] == conf_location_encoded:
            adjusted_probabilities[i] += location_boost
        
        # Check if doctor's specialization matches conference specialization
        if row['Specialization'] == conf_specialization_encoded:
            adjusted_probabilities[i] += specialization_boost
        
        # Cap the probability at 1.0
        if adjusted_probabilities[i] > 1.0:
            adjusted_probabilities[i] = 1.0
    
    # Create results
    results = []
    for i, row in prediction_data.iterrows():
        if adjusted_probabilities[i] > 0.5:  # Threshold for attendance
            results.append({
                "Doctor_ID": row['Doctor_ID'],
                "Name": row['Name'],
                "Probability": float(adjusted_probabilities[i]),
                "Location_Match": "Yes" if row['Location'] == conf_location_encoded else "No",
                "Specialization_Match": "Yes" if row['Specialization'] == conf_specialization_encoded else "No"
            })
    
    # Sort results by probability (highest first)
    results.sort(key=lambda x: x['Probability'], reverse=True)
    
    return {
        "total_doctors_predicted": len(results),
        "doctors_likely_to_attend": results,
        "conference_details": {
            "location": conf_location,
            "specialization": conf_specialization
        }
    }