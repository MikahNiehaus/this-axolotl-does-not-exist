import { useState, useEffect } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import axolotlCute from './assets/axolotl-cute.png'; // Place a cute axolotl PNG in assets

// Dynamically set API base URL based on current hostname
const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const API_BASE = isLocalhost ? 'http://192.168.1.160:5000' : import.meta.env.VITE_API_BASE || 'https://axolotl-ai-app.up.railway.app';

console.log('[Axolotl GAN Demo] Using API base URL:', API_BASE);

function App() {
  const [count, setCount] = useState(0)
  const [image, setImage] = useState(null)
  const [loading, setLoading] = useState(true)
  const [regen, setRegen] = useState(0)
  const [modelType, setModelType] = useState('GAN') // Default to GAN
  const [logs, setLogs] = useState([])
  const [error, setError] = useState(null)
  useEffect(() => {
    setLoading(true)
    setLogs([])
    setError(null)
    console.log('[Axolotl GAN Demo] Fetching:', `${API_BASE}/generate`);
    fetch(`${API_BASE}/generate`)
      .then(res => res.json())
      .then(data => {
        console.log('[Axolotl GAN Demo] Backend response:', data);
        const imgData = data.image ? `data:image/png;base64,${data.image}` : null;
        setLogs(data.logs || []);
        setError(data.error || null);
        if (data.model_type) {
          setModelType(data.model_type);
          console.log(`[Axolotl Demo] Using model type from API: ${data.model_type}`);
        } else {
          console.log(`[Axolotl Demo] Using default model type: ${modelType}`);
        }
        setImage(imgData);
        setLoading(false);
      })
      .catch((err) => {
        console.error('[Axolotl GAN Demo] Fetch error:', err);
        setError('Network or server error: ' + err.message);
        setLoading(false);
      })
  }, [regen])

  return (
    <>
      <h1>Axolotl {modelType} Generator</h1>
      {loading ? (
        <div style={{marginTop: 40}}>
          <img src={axolotlCute} alt="Axolotl Loading" style={{width: 120, animation: 'axolotl-bounce 1.2s infinite'}} />
          <div style={{fontSize: 22, marginTop: 16, color: '#b48be4', fontWeight: 'bold'}}>
            Generating a new axolotl for you...
          </div>
        </div>      ) : (
        image && (
          <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 32}}>
            <div style={{
              background: '#724b9b',
              color: 'white',
              padding: '4px 12px',
              borderRadius: '12px',
              fontWeight: 'bold',
              marginBottom: '8px'
            }}>
              Generated with {modelType} Model
            </div>
            <img
              src={image}
              alt="Generated Axolotl"
              style={{
                width: 384,
                height: 384,
                objectFit: 'contain',
                border: '2px solid #ccc',
                borderRadius: 16,
                background: '#fff',
                boxShadow: '0 4px 24px #0001',
                marginBottom: 16
              }}
            />
            <button
              style={{
                padding: '10px 24px',
                fontSize: 18,
                background: '#b48be4',
                color: '#fff',
                border: 'none',
                borderRadius: 8,
                cursor: 'pointer',
                marginTop: 8
              }}
              onClick={() => {
                const link = document.createElement('a');
                link.href = image;
                link.download = 'axolotl.png';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
              }}
            >
              Download Image
            </button>
            <button
              style={{
                padding: '10px 24px',
                fontSize: 18,
                background: '#7ed6df',
                color: '#222',
                border: 'none',
                borderRadius: 8,
                cursor: 'pointer',
                marginTop: 8,
                marginLeft: 8
              }}
              onClick={() => setRegen(r => r + 1)}
            >
              Generate New Axolotl
            </button>
            {/* Show backend logs and errors */}
            {(logs.length > 0 || error) && (
              <div style={{
                marginTop: 24,
                width: 420,
                background: '#f8f6ff',
                border: '1px solid #b48be4',
                borderRadius: 10,
                padding: 16,
                color: '#4b2e83',
                fontFamily: 'monospace',
                fontSize: 14,
                textAlign: 'left',
                boxShadow: '0 2px 8px #b48be433'
              }}>
                <div style={{fontWeight: 'bold', marginBottom: 6}}>Backend Logs:</div>
                {logs.map((line, idx) => (
                  <div key={idx}>{line}</div>
                ))}
                {error && (
                  <div style={{color: '#c0392b', marginTop: 8, fontWeight: 'bold'}}>Error: {error}</div>
                )}
              </div>
            )}
          </div>
        )
      )}
    </>
  )
}

export default App
