import React, { useState } from 'react';
import './App.css';

function App() {
  const [idea, setIdea] = useState('');
  const [theme, setTheme] = useState('');
  const [names, setNames] = useState([]);
  const [selectedName, setSelectedName] = useState('');
  const [logoUrl, setLogoUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!idea.trim() || !theme.trim()) {
      setError('Please enter both business idea and theme');
      return;
    }

    setLoading(true);
    setNames([]);
    setLogoUrl('');
    setError('');
    setSelectedName('');

    try {
      const response = await fetch('http://localhost:5000/generate_business_names', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ idea: idea.trim(), theme: theme.trim() }),
      });

      const data = await response.json();
      
      if (response.ok) {
        setNames(data.business_names || []);
      } else {
        setError(data.error || 'Failed to generate names. Please try again.');
      }
    } catch (err) {
      setError('Cannot connect to server. Please ensure the backend is running.');
      console.error('Error:', err);
    }
    setLoading(false);
  };

  const generateLogo = async (name) => {
    setSelectedName(name);
    setLogoUrl('');
    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:5000/generate_logo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name }),
      });

      if (response.ok) {
        const data = await response.json();
        
        if (data.business_logo_url) {
          setLogoUrl(data.business_logo_url);
        } else {
          setError('No logo URL received from server');
        }
      } else {
        setError(`Failed to generate logo: ${response.status}`);
      }
    } catch (err) {
      console.error('Logo generation error:', err);
      setError('Cannot connect to server');
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🚀 AI Business Generator</h1>
        <p>Create unique business names and professional logos with machine learning</p>
        
        <form className="main-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="idea">Business Idea</label>
            <input
              type="text"
              id="idea"
              value={idea}
              onChange={(e) => setIdea(e.target.value)}
              placeholder="e.g., eco-friendly coffee shop, tech startup, online marketplace"
              required
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="theme">Business Theme</label>
            <input
              type="text"
              id="theme"
              value={theme}
              onChange={(e) => setTheme(e.target.value)}
              placeholder="e.g., modern, professional, creative, minimalist"
              required
            />
          </div>
          
          <button type="submit" className="generate-btn" disabled={loading}>
            {loading ? '🧠 Generating Names...' : '✨ Generate Business Names'}
          </button>
        </form>

        {error && <div className="error-message">⚠️ {error}</div>}

        {names.length > 0 && (
          <div className="results">
            <h2>Select a name to generate its logo:</h2>
            <div className="names-section">
              <ul>
                {names.map((name, index) => (
                  <li key={index}>
                    <button
                      onClick={() => generateLogo(name)}
                      disabled={loading}
                    >
                      {name}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {loading && selectedName && (
          <div className="loading-message">
            🎨 Creating professional logo for "{selectedName}"...
          </div>
        )}

        {logoUrl && (
          <div className="logo-section">
            <h3>Professional Logo for "{selectedName}"</h3>
            <img src={logoUrl} alt="Generated Business Logo" className="business-logo" />
            <p style={{color: 'rgba(255,255,255,0.7)', marginTop: '16px', fontSize: '0.9rem'}}>
              Click the logo to view full size
            </p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
