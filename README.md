# DocAttend - Doctor Attendance Prediction System

![DocAttend Logo](https://img.shields.io/badge/DocAttend-v1.0-brown) ![React](https://img.shields.io/badge/React-18.2.0-blue) ![Flask](https://img.shields.io/badge/Flask-2.3.2-green)

DocAttend is a web application that predicts which doctors are likely to attend a medical conference based on historical data. The system uses a Random Forest Classifier to make predictions and provides a user-friendly interface for uploading data, training the model, and viewing results.

## Features

- Upload CSV files containing doctor information
- Train a machine learning model using the uploaded data
- Input conference details (location and specialization)
- View predictions of which doctors are likely to attend
- Aesthetic brown and beige themed UI with responsive design

## Technology Stack

### Backend
- Python
- Flask - Web framework
- Pandas - Data manipulation
- Scikit-learn - Machine learning library
- Joblib - Model persistence

### Frontend
- React - JavaScript library for building user interfaces
- Axios - HTTP client for API requests
- CSS - Styling with brown and beige theme

## Project Structure
DocAttend.ai
├── app.py # Main Flask application
├── model.py # Machine learning model logic
├── utils.py # Helper functions for data processing
├── requirements.txt # Python dependencies
├── .gitignore # Files to ignore in version control
├── README.md # Project documentation
└── frontend/ # React frontend application
├── public/
├── src/
│ ├── App.js # Main React component
│ ├── App.css # CSS styles
│ ├── index.js # Entry point
│ └── components/
│ ├── FileUpload.js # Component for file upload
│ ├── ConferenceForm.js # Component for conference details
│ └── PredictionResults.js # Component for displaying results
├── package.json
└── ...


## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Node.js 14 or higher
- npm or yarn

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/docattend.git
   cd docattend

How to Use the Application:
Upload Data: Select and upload CSV files containing doctor information.
Train Model: Train the prediction model using the uploaded data.
Predict Attendance: Enter conference details (location and specialization) and view predictions.


Expected CSV Format
Doctor_ID,Name,Specialization,Location,Email,Phone,Attended_2021,Attended_2022,Attended_2023,Experience

API Endpoints:
GET /test: Check if the backend is running
POST /upload-csv: Upload CSV files for processing
POST /train: Train the prediction model
POST /predict: Make predictions based on conference details


Machine Learning Model:
The system uses a Random Forest Classifier to predict doctor attendance. The model considers:
Doctor's location and specialization
Past attendance history
Years of experience
Match with conference location and specialization
