import React, { useState } from 'react';
const PredictionResults = ({ results, conference }) => {
  const [showAllDoctors, setShowAllDoctors] = useState(false);
  
  // Function to download results as CSV with only essential columns
  const downloadCSV = () => {
    // Create CSV content with detailed header
    let csvContent = `Conference: ${conference.specialization} in ${conference.location}\n`;
    csvContent += `Total Doctors Predicted to Attend: ${results.total_doctors_predicted}\n\n`;
    
    // Add simplified headers - only essential columns
    csvContent += "Doctor Name,Probability,Base Probability,Will Attend\n";
    
    // Add data rows - only for doctors predicted to attend
    const attendingDoctors = results.all_doctors.filter(doctor => doctor['Will Attend'] === "Yes");
    
    attendingDoctors.forEach(doctor => {
      csvContent += `"${doctor['Doctor Name']}",${doctor['Probability']},${doctor['Base Probability']},${doctor['Will Attend']}\n`;
    });
    
    // Create download link
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `attending_doctors_${conference.specialization}_${conference.location}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  // Doctors to display (all or only those predicted to attend)
  const doctorsToDisplay = showAllDoctors 
    ? results.all_doctors 
    : results.all_doctors.filter(doctor => doctor['Will Attend'] === "Yes");
    
  return (
    <div className="results-section">
      <div className="section-icon">📊</div>
      <div className="results-header">
        <h2>Prediction Results</h2>
        <button onClick={downloadCSV} className="btn btn-secondary download-btn">
          📥 Download Attending Doctors CSV
        </button>
      </div>
      
      <div className="conference-info">
        <h3>Conference: {conference.specialization} in {conference.location}</h3>
        <p>{results.total_doctors_predicted} out of {results.all_doctors.length} doctors predicted to attend</p>
        <p>Optimal Threshold: {(results.optimal_threshold * 100).toFixed(1)}%</p>
      </div>
      
      <div className="view-toggle">
        <button 
          onClick={() => setShowAllDoctors(!showAllDoctors)}
          className="btn btn-outline"
        >
          {showAllDoctors ? 'Show Only Attending' : 'Show All Doctors'}
        </button>
      </div>
      
      <div className="results-table-container">
        <table className="results-table">
          <thead>
            <tr>
              <th>Doctor Name</th>
              <th>Probability</th>
              <th>Base Probability</th>
              <th>Will Attend</th>
            </tr>
          </thead>
          <tbody>
            {doctorsToDisplay.map((doctor, index) => (
              <tr key={index} className={doctor['Will Attend'] === "Yes" ? "attending-row" : "not-attending-row"}>
                <td>{doctor['Doctor Name']}</td>
                <td>
                  <div className="probability-bar">
                    <div className="probability-fill" style={{ width: `${parseFloat(doctor['Probability']) * 100}%` }}></div>
                    <span className="probability-text">{doctor['Probability']}</span>
                  </div>
                </td>
                <td>{doctor['Base Probability']}</td>
                <td>
                  <span className={`attendance-badge ${doctor['Will Attend'] === "Yes" ? "will-attend" : "will-not-attend"}`}>
                    {doctor['Will Attend']}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      <div className="results-summary">
        <h3>How Predictions Were Made</h3>
        <p>The prediction model uses:</p>
        <ul>
          <li><strong>Base Probability:</strong> Initial prediction from the Random Forest model based on doctor's characteristics</li>
          <li><strong>Location Adjustment:</strong> +10% if doctor's location matches conference location</li>
          <li><strong>Specialization Adjustment:</strong> +15% if doctor's specialization matches conference topic</li>
          <li><strong>Attendance History:</strong> Doctors with consistent attendance history receive higher base probabilities</li>
          <li><strong>Threshold:</strong> Doctors with final probability above {(results.optimal_threshold * 100).toFixed(1)}% are predicted to attend</li>
        </ul>
        <p><strong>CSV Download includes:</strong> Only doctors predicted to attend with essential information for sharing.</p>
      </div>
    </div>
  );
};
export default PredictionResults;