import React, { useState } from 'react';
import CanvasEditor from './CanvasEditor';
import './App.css';

function App() {
  const [imageFile, setImageFile] = useState(null);

  const handleFileChange = (event) => {
    if (event.target.files && event.target.files[0]) {
      setImageFile(URL.createObjectURL(event.target.files[0]));
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>Investigator Web Tool</h1>
      </header>

      <div className="upload-container">
        <label htmlFor="file-upload" className="upload-label">
          Select an image to edit
        </label>
        <input
          id="file-upload"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="file-input"
        />
      </div>

      {imageFile && <CanvasEditor image={imageFile} />}
    </div>
  );
}

export default App;
