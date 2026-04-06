import React, { useState, useRef } from 'react';
import './App.css';

function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  
  const inputRef = useRef(null);

  const handleFile = async (file) => {
    if (!file) return;

    setPreview(URL.createObjectURL(file));
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8089/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "An error occurred during AI inference.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <>
      <nav className="navbar">
        <div className="nav-logo">
          <h2>LensAI</h2>
        </div>
        <div style={{color:'#64748b', fontSize:'0.85rem', fontWeight: 500}}>Enterprise Threat Detection</div>
      </nav>

      <div className="auth-container">
        
        {!loading && !result && (
          <>
            <div className="header-text">
              <h1>Deepfake Detection Protocol</h1>
              <p>Securely analyze microscopic spatial matrices and expose synthesized deepfake anomalies.</p>
            </div>

            <div 
              className={`upload-card ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => inputRef.current.click()}
            >
              <input 
                ref={inputRef}
                type="file" 
                accept="image/*" 
                style={{display:'none'}} 
                onChange={(e) => handleFile(e.target.files[0])} 
              />
              <div className="upload-icon">📸</div>
              <div className="upload-title">Drag & drop image here</div>
              <div className="upload-subtitle">Browse from device (JPG/PNG)</div>
            </div>
          </>
        )}

        {loading && (
          <div className="loading-container">
            <h2 style={{marginBottom: '1.5rem', color:'#fff', fontSize:'1.4rem'}}>Target Locked. Initializing XAI...</h2>
            {preview && (
              <div className="scanner-box">
                <img src={preview} alt="Scanning..." style={{width:'100%', height:'100%', objectFit:'cover', opacity: 0.6}} />
                <div className="scanner-line"></div>
              </div>
            )}
            <div className="progress-container">
              <div style={{color: '#94a3b8', fontSize: '0.95rem'}}>Parsing biological contours & extracting matrices</div>
              <div className="progress-bar-bg">
                <div className="progress-bar-fill"></div>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="status-fake result-status" style={{marginTop: '1.5rem'}}>
            <h3 style={{color: '#f87171', marginBottom:'0.5rem', fontSize:'1.2rem'}}>⚠️ Interference Detected</h3>
            <p style={{color: '#fca5a5', fontSize:'0.95rem'}}>{error}</p>
            <button onClick={() => {setError(null); setPreview(null);}} className="reset-btn" style={{marginTop: '1.2rem', background: '#374151', padding: '0.6rem 1.5rem', boxShadow:'none', fontSize:'0.95rem'}}>Dismiss Threat</button>
          </div>
        )}

        {result && (
          <div style={{animation: 'fadeUp 0.6s ease'}}>
            <div className={`result-status ${result.is_fake ? 'status-fake' : 'status-real'}`}>
              <h1 style={{color: result.is_fake ? '#f87171' : '#34d399', fontSize:'1.8rem', letterSpacing: '-0.5px'}}>
                {result.is_fake ? '🚨 SYNTHETIC DEEPFAKE VERIFIED' : '✅ AUTHENTIC BIOLOGICAL MATCH'}
              </h1>
              
              <div style={{marginTop: '1.2rem'}}>
                <span style={{fontSize:'1.1rem', color:'#cbd5e1'}}>Network Confidence Profile: </span>
                <span style={{fontSize:'1.4rem', fontWeight:'700', color: result.is_fake ? '#f87171' : '#34d399'}}>{result.confidence}%</span>
              </div>
              
              <div className="confidence-gauge">
                <div 
                  className="gauge-fill" 
                  style={{
                    width: `${result.confidence}%`, 
                    background: result.is_fake ? 'linear-gradient(90deg, #ef4444, #f87171)' : 'linear-gradient(90deg, #10b981, #34d399)'
                  }}
                ></div>
              </div>
            </div>
            
            <div className="results-grid">
               <div className="result-card">
                 <h3><span>👤</span> Uploaded Subject</h3>
                 {preview && <img src={preview} alt="Original" className="img-preview" />}
                 <p style={{marginTop:'1rem', fontSize:'0.85rem', color:'#94a3b8', lineHeight: 1.5}}>The raw, unaltered payload submitted directly by the user interface.</p>
               </div>
               
               <div className="result-card">
                 <h3><span>🎯</span> Target Mesh Isolation</h3>
                 <img src={result.face_crop} alt="Crop" className="img-preview" />
                 <p style={{marginTop:'1rem', fontSize:'0.85rem', color:'#94a3b8', lineHeight: 1.5}}>MTCNN geometry engine autonomously isolates the face, pruning boundary noise.</p>
               </div>
               
               <div className="result-card">
                 <h3><span>🔥</span> XAI Heatmap Engine</h3>
                 <img src={result.heatmap} alt="Grad-CAM" className="img-preview" />
                 <p style={{marginTop:'1rem', fontSize:'0.85rem', color:'#94a3b8', lineHeight: 1.5}}>Grad-CAM natively maps exact microscopic gradients dictating AI suspicion.</p>
               </div>
            </div>
            
            <div style={{textAlign: 'center', marginTop: '2.5rem'}}>
              <button onClick={() => {setResult(null); setPreview(null);}} className="reset-btn">Initialize New Scan Protocol</button>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

export default App;
