import { useState, useEffect } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import axolotlCute from './assets/axolotl-cute.png'; // Place a cute axolotl PNG in assets

function App() {
  const [count, setCount] = useState(0)
  const [image, setImage] = useState(null)
  const [loading, setLoading] = useState(true)
  const [regen, setRegen] = useState(0)

  useEffect(() => {
    setLoading(true)
    fetch('/generate')
      .then(res => res.json())
      .then(data => {
        setImage(`data:image/png;base64,${data.image}`)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [regen])

  return (
    <>
      <h1>Axolotl GAN Demo</h1>
      {loading ? (
        <div style={{marginTop: 40}}>
          <img src={axolotlCute} alt="Axolotl Loading" style={{width: 120, animation: 'axolotl-bounce 1.2s infinite'}} />
          <div style={{fontSize: 22, marginTop: 16, color: '#b48be4', fontWeight: 'bold'}}>
            Generating a new axolotl for you...
          </div>
        </div>
      ) : (
        image && (
          <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 32}}>
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
          </div>
        )
      )}
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Vite + React</h1>
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.jsx</code> and save to test HMR
        </p>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
    </>
  )
}

export default App
