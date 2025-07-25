// Simple Business Name & Logo Generator
import React, { useState } from 'react';
import './App.css';

function App() {
  const [idea, setIdea] = useState('');
  const [theme, setTheme] = useState('');
  const [names, setNames] = useState([]);
  const [selectedName, setSelectedName] = useState('');
  const [logoUrl, setLogoUrl] = useState('');
  const [loadingNames, setLoadingNames] = useState(false);
  const [loadingLogo, setLoadingLogo] = useState(false);
  const [error, setError] = useState('');
  const [connectionStatus, setConnectionStatus] = useState('');

  const testConnection = async () => {
    setConnectionStatus('Testing...');
    try {
      const response = await fetch('http://localhost:5000/health', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        setConnectionStatus('✅ Server connection OK');
        console.log('Health check successful:', data);
      } else {
        setConnectionStatus('❌ Server responded with error');
      }
    } catch (err) {
      setConnectionStatus('❌ Cannot reach server');
      console.error('Connection test failed:', err);
    }
  };

  const generateNames = async (e) => {
    e.preventDefault();
    if (!idea.trim() || !theme.trim()) {
      setError('Please enter both business idea and theme');
      return;
    }

    setLoadingNames(true);
    setNames([]);
    setSelectedName('');
    setLogoUrl('');
    setError('');

    try {
      const response = await fetch('http://localhost:5000/generate_business_names', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ idea: idea.trim(), theme: theme.trim() }),
      });

      const data = await response.json();

      if (response.ok) {
        setNames(data.business_names || []);
        if (data.business_names && data.business_names.length === 0) {
          setError('No names generated. Try different keywords.');
        }
      } else {
        setError(data.error || 'Failed to generate names');
      }
    } catch (err) {
      setError('Cannot connect to server. Make sure the backend is running on port 5000.');
      console.error('Error:', err);
    } finally {
      setLoadingNames(false);
    }
  };

  const generateLogo = async (name) => {
    setSelectedName(name);
    setLoadingLogo(true);
    setLogoUrl('');
    setError('');

    try {
      console.log('Attempting to generate logo for:', name);
      
      // Add timeout to prevent hanging requests
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
      
      const response = await fetch('http://localhost:5000/generate_logo', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      console.log('Response status:', response.status);
      console.log('Response ok:', response.ok);

      const data = await response.json();
      console.log('Response data:', data);

      if (response.ok) {
        setLogoUrl(data.business_logo_url || '');
        if (!data.business_logo_url) {
          setError('Logo generated but no URL returned');
        }
      } else {
        setError(data.error || 'Failed to generate logo');
      }
    } catch (err) {
      console.error('Logo generation error:', err);
      if (err.name === 'TypeError' && err.message.includes('Failed to fetch')) {
        setError('🔌 Connection failed! Please check: 1) Backend server is running, 2) No firewall blocking localhost:5000, 3) Try the "Test Server Connection" button above.');
      } else {
        setError('Cannot connect to server for logo generation. Please ensure the backend is running on port 5000.');
      }
      console.error('Logo error:', err);
    } finally {
      setLoadingLogo(false);
    }
  };

  return (
    <div className="App">
      <div className="App-header">
        <h1>🚀 Business Name & Logo Generator</h1>
        <p>Enter your business idea and theme to generate creative names and logos</p>
        
        <div style={{ marginBottom: '20px' }}>
          <button 
            onClick={testConnection} 
            style={{ 
              background: '#28a745', 
              color: 'white', 
              border: 'none', 
              padding: '8px 16px', 
              borderRadius: '4px', 
              cursor: 'pointer',
              marginRight: '10px'
            }}
          >
            Test Server Connection
          </button>
          {connectionStatus && <span style={{ color: connectionStatus.includes('✅') ? '#28a745' : '#dc3545' }}>{connectionStatus}</span>}
        </div>
        
        <form onSubmit={generateNames} className="main-form">
          <div className="form-group">
            <label>Business Idea:</label>
            <input
              type="text"
              value={idea}
              onChange={(e) => setIdea(e.target.value)}
              placeholder="e.g., coffee shop, tech startup, bakery"
              required
            />
          </div>
          
          <div className="form-group">
            <label>Theme:</label>
            <input
              type="text"
              value={theme}
              onChange={(e) => setTheme(e.target.value)}
              placeholder="e.g., modern, eco-friendly, vintage"
              required
            />
          </div>
          
          <button type="submit" disabled={loadingNames} className="generate-btn">
            {loadingNames ? '🔄 Generating Names...' : 'Generate Business Names'}
          </button>
        </form>

        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}

        {names.length > 0 && (
          <div className="results-section">
            <h2>🎉 Generated Names (Click to generate logo):</h2>
            <div className="names-grid">
              {names.map((name, index) => (
                <button
                  key={index}
                  className={`name-card ${selectedName === name ? 'selected' : ''}`}
                  onClick={() => generateLogo(name)}
                  disabled={loadingLogo}
                >
                  {name}
                </button>
              ))}
            </div>
          </div>
        )}

        {loadingLogo && (
          <div className="loading-message">
            🎨 Generating logo for "{selectedName}"...
          </div>
        )}

        {logoUrl && selectedName && (
          <div className="logo-section">
            <h2>🎨 Logo for "{selectedName}"</h2>
            <div className="logo-container">
              <img src={logoUrl} alt={`${selectedName} logo`} className="logo-image" />
              <div className="logo-actions">
                <a href={logoUrl} download={`${selectedName.replace(/\s+/g, '_')}_logo.svg`} className="download-btn">
                  💾 Download Logo
                </a>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;