import React from 'react';

function App() {
  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      color: 'white',
      fontSize: '24px',
      textAlign: 'center'
    }}>
      <div>
        <h1>🚀 Business Generator</h1>
        <p>If you can see this, React is working!</p>
        <button style={{
          padding: '15px 30px',
          fontSize: '18px',
          background: '#ff6b6b',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          cursor: 'pointer'
        }}>
          Test Button
        </button>
      </div>
    </div>
  );
}

export default App;
