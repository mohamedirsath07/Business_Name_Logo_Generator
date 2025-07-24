// src/App.js (React Frontend)
import React, { useState } from 'react';
import './App.css'; // Basic CSS for styling

function App() {
  const [idea, setIdea] = useState('');
  const [theme, setTheme] = useState('');
  const [names, setNames] = useState([]);
  const [selectedName, setSelectedName] = useState('');
  const [logoUrl, setLogoUrl] = useState('');
  const [loadingNames, setLoadingNames] = useState(false);
  const [loadingLogo, setLoadingLogo] = useState(false);
  const [error, setError] = useState('');

  const handleNameSubmit = async (e) => {
    e.preventDefault();
    setLoadingNames(true);
    setNames([]);
    setLogoUrl('');
    setError('');
    setSelectedName('');

    try {
      const response = await fetch('http://localhost:5000/generate_business_names', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ idea, theme }),
      });

      const data = await response.json();

      if (response.ok) {
        setNames(data.business_names || []);
      } else {
        setError(data.error || 'An unknown error occurred.');
      }
    } catch (err) {
      setError('Failed to connect to the backend server. Please try again later.');
      console.error('Fetch error:', err);
    } finally {
      setLoadingNames(false);
    }
  };

  const handleNameClick = async (name) => {
    setSelectedName(name);
    setLoadingLogo(true);
    setLogoUrl('');
    setError('');

    try {
      const response = await fetch('http://localhost:5000/generate_logo', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name }),
      });

      const data = await response.json();

      if (response.ok) {
        setLogoUrl(data.business_logo_url || '');
      } else {
        setError(data.error || 'An unknown error occurred.');
      }
    } catch (err) {
      setError('Failed to connect to the backend server. Please try again later.');
      console.error('Fetch error:', err);
    } finally {
      setLoadingLogo(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Business Name & Logo Generator</h1>
        <form onSubmit={handleNameSubmit}>
          <div className="form-group">
            <label htmlFor="idea">Business Idea:</label>
            <input
              type="text"
              id="idea"
              value={idea}
              onChange={(e) => setIdea(e.target.value)}
              placeholder="e.g., eco-friendly coffee shop"
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="theme">Business Theme:</label>
            <input
              type="text"
              id="theme"
              value={theme}
              onChange={(e) => setTheme(e.target.value)}
              placeholder="e.g., minimalist, rustic, futuristic"
              required
            />
          </div>
          <button type="submit" disabled={loadingNames}>
            {loadingNames ? 'Generating Names...' : 'Generate Names'}
          </button>
        </form>

        {error && <p className="error-message">{error}</p>}

        {!loadingNames && names.length > 0 && (
          <div className="results">
            <h2>Choose a Name to Generate a Logo:</h2>
            <div className="names-section">
              <ul>
                {names.map((name, index) => (
                  <li key={index}>
                    <button onClick={() => handleNameClick(name)} disabled={loadingLogo}>
                      {name}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {loadingLogo && <p>Generating logo for "{selectedName}"...</p>}

        {logoUrl && (
          <div className="logo-section">
            <h3>Business Logo for "{selectedName}":</h3>
            <img src={logoUrl} alt="Generated Business Logo" className="business-logo" />
          </div>
        )}
      </header>
    </div>
  );
}

export default App;