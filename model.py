import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_advanced_features(data):
    """Create comprehensive features for maximum accuracy"""
    features = data.copy()
    
    # Basic attendance features
    features['Total_Attended'] = data[['Attended_2021', 'Attended_2022', 'Attended_2023']].sum(axis=1)
    features['Attendance_Rate'] = features['Total_Attended'] / 3
    
    # Recency and frequency features
    features['Years_Since_Last_Attendance'] = data[['Attended_2021', 'Attended_2022', 'Attended_2023']].apply(
        lambda row: 3 - (row.values[::-1].argmax() + 1) if row.max() == 1 else 3, axis=1)
    
    # Consistency and trend features
    features['Attendance_Consistency'] = data[['Attended_2021', 'Attended_2022', 'Attended_2023']].std(axis=1)
    features['Attendance_Trend'] = data['Attended_2023'] - data['Attended_2021']
    features['Attendance_Pattern'] = data[['Attended_2021', 'Attended_2022', 'Attended_2023']].apply(
        lambda row: ''.join(row.astype(str)), axis=1)
    
    # Experience transformations
    if 'Experience' in features.columns:
        features['Experience_Squared'] = data['Experience'] ** 2
        features['Experience_Log'] = np.log1p(data['Experience'])
        features['Experience_Sqrt'] = np.sqrt(data['Experience'])
        features['Experience_Attendance_Interaction'] = data['Experience'] * features['Attendance_Rate']
    
    # Location and specialization features with more granularity
    location_stats = data.groupby('Location')['Attended_2023'].agg(['mean', 'count', 'std'])
    specialization_stats = data.groupby('Specialization')['Attended_2023'].agg(['mean', 'count', 'std'])
    
    features['Location_Attendance_Rate'] = data['Location'].map(location_stats['mean']).fillna(0.5)
    features['Location_Popularity'] = data['Location'].map(location_stats['count']).fillna(1)
    features['Location_Consistency'] = data['Location'].map(location_stats['std']).fillna(0)
    
    features['Specialization_Attendance_Rate'] = data['Specialization'].map(specialization_stats['mean']).fillna(0.5)
    features['Specialization_Popularity'] = data['Specialization'].map(specialization_stats['count']).fillna(1)
    features['Specialization_Consistency'] = data['Specialization'].map(specialization_stats['std']).fillna(0)
    
    # Interaction features
    features['Location_Specialization_Interaction'] = features['Location_Attendance_Rate'] * features['Specialization_Attendance_Rate']
    if 'Experience' in features.columns:
        features['Location_Experience_Interaction'] = features['Location_Attendance_Rate'] * data['Experience']
        features['Specialization_Experience_Interaction'] = features['Specialization_Attendance_Rate'] * data['Experience']
    
    # Advanced pattern features
    features['Consistent_Attender'] = (features['Total_Attended'] >= 2).astype(int)
    features['Recent_Attender'] = (data['Attended_2023'] == 1).astype(int)
    features['Improving_Attendance'] = ((data['Attended_2023'] > data['Attended_2022']) & 
                                      (data['Attended_2022'] > data['Attended_2021'])).astype(int)
    
    # Fill any missing values
    features.fillna(0, inplace=True)
    
    return features

def train_model(processed_data, encoders):
    """Train optimized Random Forest model with hyperparameter tuning"""
    logger.info("Training optimized Random Forest model")
    
    # Create advanced features
    features = create_advanced_features(processed_data)
    
    # Define features and target
    X = features.drop(['Attended_2023', 'Doctor_ID', 'Name', 'Email', 'Phone'], axis=1, errors='ignore')
    y = features['Attended_2023']
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    
    # Define parameter distribution for RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True],
        'class_weight': ['balanced'],
        'criterion': ['gini']
    }
    
    # Create Random Forest model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # RandomizedSearchCV for efficient hyperparameter tuning
    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Fit the model
    rf_search.fit(X_train, y_train)
    
    # Get the best model
    best_rf = rf_search.best_estimator_
    
    # Make predictions
    y_pred = best_rf.predict(X_test)
    y_proba = best_rf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    # Find optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    youden_j = tpr - fpr
    optimal_threshold = thresholds[np.argmax(youden_j)]
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Create model data
    model_data = {
        'model': best_rf,
        'feature_names': X.columns.tolist(),
        'optimal_threshold': optimal_threshold,
        'feature_importance': feature_importance.to_dict(),
        'model_metrics': {
            'accuracy': accuracy,
            'auc': auc,
            'best_params': rf_search.best_params_
        },
        'encoders': encoders
    }
    
    # Save model
    joblib.dump(model_data, 'optimized_rf_model.joblib')
    
    logger.info(f"Model trained successfully. Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    
    return model_data

def predict_attendance(data, model_data, conf_location, conf_specialization):
    logger.info(f"Making predictions for {conf_specialization} conference in {conf_location}")
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    optimal_threshold = model_data['optimal_threshold']
    encoders = model_data['encoders']
    
    # Make a copy of the data to avoid modifying the original
    data_copy = data.copy()
    
    # Store original location and specialization for matching
    original_locations = data_copy['Location'].copy()
    original_specializations = data_copy['Specialization'].copy()
    
    # Preprocess the data for prediction: encode categorical variables
    # Handle missing values
    data_copy.fillna({
        'Specialization': 'Unknown',
        'Location': 'Unknown',
        'Experience': data_copy['Experience'].median() if 'Experience' in data_copy.columns else 5
    }, inplace=True)
    
    # Encode categorical variables using the same encoders from training
    for col in ['Location', 'Specialization']:
        if col in data_copy.columns:
            # Handle unknown categories by assigning a special value (-1)
            known_categories = set(encoders[col].classes_)
            data_copy[col] = data_copy[col].apply(lambda x: encoders[col].transform([x])[0] if x in known_categories else -1)
    
    # Now create features using the preprocessed data
    features = create_advanced_features(data_copy)
    
    # Create masks for matches - ROBUST APPROACH
    location_mask = []
    specialization_mask = []
    
    # Clean conference location and specialization for comparison
    clean_conf_location = conf_location.strip().lower()
    clean_conf_specialization = conf_specialization.strip().lower()
    
    logger.info(f"Conference location: '{clean_conf_location}', Conference specialization: '{clean_conf_specialization}'")
    
    # Log first 5 doctors for debugging
    logger.info("First 5 doctors:")
    for i in range(min(5, len(data))):
        doc_location = original_locations.iloc[i]
        doc_specialization = original_specializations.iloc[i]
        logger.info(f"  Doctor {i}: Location='{doc_location}' (type: {type(doc_location)}), "
                   f"Specialization='{doc_specialization}' (type: {type(doc_specialization)})")
    
    for i in range(len(data)):
        # Get doctor's location and specialization
        doc_location = original_locations.iloc[i]
        doc_specialization = original_specializations.iloc[i]
        
        # Clean doctor's location and specialization for comparison
        # Convert to string and remove any non-printable characters
        clean_doc_location = str(doc_location).strip().lower()
        clean_doc_specialization = str(doc_specialization).strip().lower()
        
        # Remove any non-alphanumeric characters except spaces
        clean_doc_location = ''.join(c for c in clean_doc_location if c.isalnum() or c.isspace())
        clean_doc_specialization = ''.join(c for c in clean_doc_specialization if c.isalnum() or c.isspace())
        
        # Remove extra spaces
        clean_doc_location = ' '.join(clean_doc_location.split())
        clean_doc_specialization = ' '.join(clean_doc_specialization.split())
        
        # Check for matches
        loc_match = (clean_doc_location == clean_conf_location)
        spec_match = (clean_doc_specialization == clean_conf_specialization)
        
        location_mask.append(loc_match)
        specialization_mask.append(spec_match)
        
        # Log first 5 doctors with detailed comparison
        if i < 5:
            logger.info(f"  Doctor {i} comparison:")
            logger.info(f"    Cleaned location: '{clean_doc_location}' vs '{clean_conf_location}' -> {loc_match}")
            logger.info(f"    Cleaned specialization: '{clean_doc_specialization}' vs '{clean_conf_specialization}' -> {spec_match}")
    
    # Convert to numpy arrays
    location_mask = np.array(location_mask)
    specialization_mask = np.array(specialization_mask)
    
    # Log match statistics
    logger.info(f"Location matches: {location_mask.sum()}/{len(location_mask)}")
    logger.info(f"Specialization matches: {specialization_mask.sum()}/{len(specialization_mask)}")
    
    # Ensure all required features exist
    for col in feature_names:
        if col not in features.columns:
            features[col] = 0
    
    # Select features in correct order
    X = features[feature_names]
    
    # Get base predictions
    probabilities = model.predict_proba(X)[:, 1]
    
    # Apply conference-specific adjustments
    location_importance = 0.20  # +20% for location match
    specialization_importance = 0.25  # +25% for specialization match
    location_penalty = 0.05  # -5% for location mismatch
    specialization_penalty = 0.08  # -8% for specialization mismatch
    
    # Additional boost for doctors with both matches
    both_matches_bonus = 0.15  # +15% for having both location and specialization match
    
    adjusted_probabilities = probabilities.copy()
    
    # Apply adjustments for matches
    adjusted_probabilities[location_mask] += location_importance
    adjusted_probabilities[specialization_mask] += specialization_importance
    
    # Apply additional bonus for doctors with both matches
    both_matches = location_mask & specialization_mask
    adjusted_probabilities[both_matches] += both_matches_bonus
    
    # Apply penalties for non-matches
    adjusted_probabilities[~location_mask] -= location_penalty
    adjusted_probabilities[~specialization_mask] -= specialization_penalty
    
    # Cap probabilities between 0 and 1
    adjusted_probabilities = np.clip(adjusted_probabilities, 0, 1.0)
    
    # Use a more reasonable threshold
    selective_threshold = max(optimal_threshold, 0.55)  # Reduced from 0.60 to 0.55
    
    # Make final predictions
    predictions = (adjusted_probabilities >= selective_threshold).astype(int)
    
    # ENSURE AT LEAST SOME DOCTORS ARE PREDICTED TO ATTEND
    # If no doctors meet the threshold, take the top 15%
    if predictions.sum() == 0:
        logger.warning("No doctors meet threshold. Selecting top 15% by probability.")
        sorted_indices = np.argsort(adjusted_probabilities)[::-1]
        top_15_percent = int(len(data) * 0.15)
        if top_15_percent == 0:
            top_15_percent = 1
        predictions = np.zeros(len(data), dtype=int)
        predictions[sorted_indices[:top_15_percent]] = 1
    
    # Special rule for doctors with both matches and strong past attendance
    # Identify doctors with both matches
    for i, row in data.iterrows():
        if both_matches[i]:
            # Check if they attended at least 2 out of 3 previous conferences
            if all(col in row for col in ['Attended_2021', 'Attended_2022', 'Attended_2023']):
                total_attended = sum([row['Attended_2021'], row['Attended_2022'], row['Attended_2023']])
                if total_attended >= 2:
                    # Force prediction to attend for these doctors
                    predictions[i] = 1
                    logger.info(f"Overriding prediction for doctor {row['Doctor_ID']} ({row['Name']}) - both matches and strong attendance history")
    
    # Create detailed results (for backend use)
    results = []
    for i, row in data.iterrows():
        result = {
            "Doctor_ID": row['Doctor_ID'],
            "Name": row['Name'],
            "Probability": float(adjusted_probabilities[i]),
            "Base_Probability": float(probabilities[i]),
            "Location_Match": "Yes" if location_mask[i] else "No",
            "Specialization_Match": "Yes" if specialization_mask[i] else "No",
            "Both_Matches": "Yes" if both_matches[i] else "No",
            "Location_Adjustment": float(location_importance) if location_mask[i] else float(-location_penalty),
            "Specialization_Adjustment": float(specialization_importance) if specialization_mask[i] else float(-specialization_penalty),
            "Both_Matches_Bonus": float(both_matches_bonus) if both_matches[i] else 0.0,
            "Experience": float(row['Experience']) if 'Experience' in row else 0.0,
            "Location": str(original_locations.iloc[i]) if 'Location' in row else "Unknown",
            "Specialization": str(original_specializations.iloc[i]) if 'Specialization' in row else "Unknown",
            "Will_Attend": "Yes" if predictions[i] == 1 else "No",
            "Threshold": float(selective_threshold)
        }
        
        # Add attendance history if available
        if all(col in row for col in ['Attended_2021', 'Attended_2022', 'Attended_2023']):
            result["Attendance_History"] = {
                "2021": int(row['Attended_2021']),
                "2022": int(row['Attended_2022']),
                "2023": int(row['Attended_2023'])
            }
            
            # Calculate attendance rate
            total_attended = sum([row['Attended_2021'], row['Attended_2022'], row['Attended_2023']])
            result["Attendance_Rate"] = float(total_attended / 3)
        
        # Add detailed prediction explanation
        explanation_parts = []
        
        # Base probability analysis
        if probabilities[i] >= 0.7:
            explanation_parts.append("High base probability")
        elif probabilities[i] >= 0.5:
            explanation_parts.append("Moderate base probability")
        else:
            explanation_parts.append("Low base probability")
        
        # Match impacts
        if location_mask[i]:
            explanation_parts.append(f"Location match (+{location_importance:.1%})")
        else:
            explanation_parts.append(f"Location mismatch (-{location_penalty:.1%})")
            
        if specialization_mask[i]:
            explanation_parts.append(f"Specialization match (+{specialization_importance:.1%})")
        else:
            explanation_parts.append(f"Specialization mismatch (-{specialization_penalty:.1%})")
        
        # Both matches bonus
        if both_matches[i]:
            explanation_parts.append(f"Both matches bonus (+{both_matches_bonus:.1%})")
        
        # Final decision
        if predictions[i] == 1:
            if adjusted_probabilities[i] - selective_threshold > 0.15:
                explanation_parts.append("Well above threshold")
            else:
                explanation_parts.append("Above threshold")
        else:
            if selective_threshold - adjusted_probabilities[i] > 0.15:
                explanation_parts.append("Well below threshold")
            else:
                explanation_parts.append("Below threshold")
        
        result["Explanation"] = "; ".join(explanation_parts)
        
        results.append(result)
    
    # Sort by probability
    results.sort(key=lambda x: x['Probability'], reverse=True)
    
    # Create simplified results for return (only the four columns)
    simplified_results = []
    for r in results:
        simplified_results.append({
            'Doctor Name': r['Name'],
            'Probability': f"{r['Probability']:.1%}",
            'Base Probability': f"{r['Base_Probability']:.1%}",
            'Will Attend': r['Will_Attend']
        })
    
    # Return only the results for display
    return {
        "total_doctors_predicted": len([r for r in simplified_results if r["Will Attend"] == "Yes"]),
        "all_doctors": simplified_results,  # Now only contains the four columns
        "model_metrics": model_data['model_metrics'],
        "optimal_threshold": float(optimal_threshold),
        "selective_threshold": float(selective_threshold),
        "conference_details": {
            "location": conf_location,
            "specialization": conf_specialization
        }
    }