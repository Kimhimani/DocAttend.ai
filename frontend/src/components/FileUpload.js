import React from 'react';

const FileUpload = ({ files, onFileChange, onUpload, loading }) => {
  return (
    <div className="file-upload-section">
      <div className="section-icon">📁</div>
      <h2>Upload Doctor Data</h2>
      <p>Upload CSV files containing doctor information</p>
      
      <div className="file-input-container">
        <input 
          type="file" 
          id="file-upload" 
          multiple 
          accept=".csv" 
          onChange={onFileChange}
          className="file-input"
        />
        <label htmlFor="file-upload" className="file-label">
          Choose CSV Files
        </label>
      </div>
      
      {files.length > 0 && (
        <div className="file-list">
          <h4>Selected Files:</h4>
          <ul>
            {Array.from(files).map((file, index) => (
              <li key={index}>{file.name}</li>
            ))}
          </ul>
        </div>
      )}
      
      <button 
        onClick={onUpload} 
        disabled={loading || files.length === 0}
        className="btn btn-primary"
      >
        {loading ? 'Uploading...' : 'Upload Files'}
      </button>
      
      <div className="sample-format">
        <h4>Expected CSV Format:</h4>
        <pre>
          Doctor_ID,Name,Specialization,Location,Email,Phone,Attended_2021,Attended_2022,Attended_2023,Experience
        </pre>
      </div>
    </div>
  );
};

export default FileUpload;