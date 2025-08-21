import React, { useState, useEffect } from 'react';
import './App.css';
import FileUpload from './components/FileUpload';
import ConferenceForm from './components/ConferenceForm';
import PredictionResults from './components/PredictionResults';
import axios from 'axios';

function App() {
  const [step, setStep] = useState(1); // 1: Upload, 2: Train, 3: Predict
  const [files, setFiles] = useState([]);
  const [conference, setConference] = useState({
    location: '',
    specialization: ''
  });
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Check backend connection on component mount
  useEffect(() => {
    checkBackendConnection();
  }, []);

  const checkBackendConnection = async () => {
    try {
      await axios.get('http://localhost:5001/test');
    } catch (err) {
      setError('Cannot connect to backend. Please make sure the Flask server is running on http://localhost:5001');
    }
  };

  // Handle file selection
  const handleFileChange = (e) => {
    setFiles(e.target.files);
  };

  // Handle conference input changes
  const handleConferenceChange = (e) => {
    const { name, value } = e.target;
    setConference(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // Upload CSV files
  const handleUpload = async () => {
    if (files.length === 0) {
      setError('Please select at least one CSV file');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    try {
      const response = await axios.post('http://localhost:5001/upload-csv', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 30000
      });

      if (response.data.status === 'success') {
        setStep(2);
      } else {
        setError(response.data.message || 'Upload failed');
      }
    } catch (err) {
      if (err.response) {
        setError(`Server error: ${err.response.data.message || err.response.statusText}`);
      } else if (err.request) {
        setError('No response from server. Please check if the backend is running.');
      } else {
        setError('Error uploading files: ' + err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  // Train the model
  const handleTrain = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:5001/train');
      
      if (response.data.status === 'success') {
        setStep(3);
      } else {
        setError(response.data.message || 'Training failed');
      }
    } catch (err) {
      if (err.response) {
        setError(`Server error: ${err.response.data.message || err.response.statusText}`);
      } else if (err.request) {
        setError('No response from server. Please check if the backend is running.');
      } else {
        setError('Error training model: ' + err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  // Make predictions
  const handlePredict = async () => {
    if (!conference.location || !conference.specialization) {
      setError('Please enter both location and specialization');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:5001/predict', {
        location: conference.location,
        specialization: conference.specialization
      });

      if (response.data.status === 'success') {
        setResults(response.data.data);
      } else {
        setError(response.data.message || 'Prediction failed');
      }
    } catch (err) {
      if (err.response) {
        setError(`Server error: ${err.response.data.message || err.response.statusText}`);
      } else if (err.request) {
        setError('No response from server. Please check if the backend is running.');
      } else {
        setError('Error making prediction: ' + err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  // Reset the process
  const handleReset = () => {
    setStep(1);
    setFiles([]);
    setConference({ location: '', specialization: '' });
    setResults(null);
    setError(null);
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="logo-container">
          <h1 className="app-title">DocAttend</h1>
          <div className="subtitle">Medical Conference Attendance Predictor</div>
        </div>
        <div className="header-decoration"></div>
      </header>

      <div className="container">
        {/* Progress Indicator */}
        <div className="progress-indicator">
          <div className={`step ${step >= 1 ? 'active' : ''}`}>
            <div className="step-number">1</div>
            <div className="step-label">Upload Data</div>
          </div>
          <div className="progress-line"></div>
          <div className={`step ${step >= 2 ? 'active' : ''}`}>
            <div className="step-number">2</div>
            <div className="step-label">Train Model</div>
          </div>
          <div className="progress-line"></div>
          <div className={`step ${step >= 3 ? 'active' : ''}`}>
            <div className="step-number">3</div>
            <div className="step-label">Predict</div>
          </div>
        </div>

        {/* Error Display */}
        {error && <div className="error-message">{error}</div>}

        {/* Step 1: File Upload */}
        {step === 1 && (
          <FileUpload
            files={files}
            onFileChange={handleFileChange}
            onUpload={handleUpload}
            loading={loading}
          />
        )}

        {/* Step 2: Train Model */}
        {step === 2 && (
          <div className="train-section">
            <div className="section-icon">🧠</div>
            <h2>Train Prediction Model</h2>
            <p>Your data has been uploaded. Now train the model to make predictions.</p>
            <button 
              onClick={handleTrain} 
              disabled={loading}
              className="btn btn-primary"
            >
              {loading ? 'Training...' : 'Train Model'}
            </button>
          </div>
        )}

        {/* Step 3: Prediction */}
        {step === 3 && (
          <>
            <ConferenceForm
              conference={conference}
              onChange={handleConferenceChange}
              onPredict={handlePredict}
              loading={loading}
            />
            
            {results && (
              <PredictionResults results={results} conference={conference} />
            )}
          </>
        )}

        {/* Reset Button */}
        {step > 1 && (
          <div className="reset-section">
            <button onClick={handleReset} className="btn btn-secondary">
              Start Over
            </button>
          </div>
        )}
      </div>
      
      <footer className="App-footer">
        <p>DocAttend © {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
}

export default App;