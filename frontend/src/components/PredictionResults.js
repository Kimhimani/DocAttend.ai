import React from 'react';

const PredictionResults = ({ results, conference }) => {
  return (
    <div className="results-section">
      <div className="section-icon">📊</div>
      <h2>Prediction Results</h2>
      <div className="conference-info">
        <h3>Conference: {conference.specialization} in {conference.location}</h3>
        <p>{results.total_doctors_predicted} doctors likely to attend</p>
      </div>
      
      <div className="results-table-container">
        <table className="results-table">
          <thead>
            <tr>
              <th>Doctor Name</th>
              <th>Probability</th>
              <th>Location Match</th>
              <th>Specialization Match</th>
            </tr>
          </thead>
          <tbody>
            {results.doctors_likely_to_attend.map((doctor, index) => (
              <tr key={index}>
                <td>{doctor.Name}</td>
                <td>{(doctor.Probability * 100).toFixed(1)}%</td>
                <td>{doctor.Location_Match}</td>
                <td>{doctor.Specialization_Match}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      <div className="results-summary">
        <h3>Summary</h3>
        <p>The prediction model considered:</p>
        <ul>
          <li>Doctor's location and specialization</li>
          <li>Past attendance history</li>
          <li>Years of experience</li>
          <li>Match with conference location and specialization</li>
        </ul>
      </div>
    </div>
  );
};

export default PredictionResults;