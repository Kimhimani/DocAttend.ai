import React from 'react';

const ConferenceForm = ({ conference, onChange, onPredict, loading }) => {
  return (
    <div className="conference-form-section">
      <div className="section-icon">📅</div>
      <h2>Conference Details</h2>
      <p>Enter details about the upcoming conference</p>
      
      <div className="form-group">
        <label htmlFor="location">Conference Location</label>
        <input
          type="text"
          id="location"
          name="location"
          value={conference.location}
          onChange={onChange}
          placeholder="e.g., Bangalore"
          className="form-control"
        />
      </div>
      
      <div className="form-group">
        <label htmlFor="specialization">Conference Specialization</label>
        <input
          type="text"
          id="specialization"
          name="specialization"
          value={conference.specialization}
          onChange={onChange}
          placeholder="e.g., Orthopedics"
          className="form-control"
        />
      </div>
      
      <button 
        onClick={onPredict} 
        disabled={loading || !conference.location || !conference.specialization}
        className="btn btn-primary"
      >
        {loading ? 'Predicting...' : 'Predict Attendance'}
      </button>
    </div>
  );
};

export default ConferenceForm;