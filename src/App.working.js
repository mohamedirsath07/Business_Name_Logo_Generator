import React, { useState } from 'react';

function App() {
  const [idea, setIdea] = useState('');
  const [theme, setTheme] = useState('');
  const [names, setNames] = useState([]);
  const [selectedName, setSelectedName] = useState('');
  const [logoUrl, setLogoUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const generateNames = async (e) => {
    e.preventDefault();
    if (!idea.trim() || !theme.trim()) {
      setError('Please enter both business idea and theme');
      return;
    }

    setLoading(true);
    setNames([]);
    setError('');

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
        setError(data.error || 'Failed to generate names');
      }
    } catch (err) {
      setError('Cannot connect to server');
    }
    setLoading(false);
  };

  const generateLogo = async (name) => {
    setSelectedName(name);
    setLogoUrl('');
    setLoading(true);
    setError('');

    try {
      console.log('Generating logo for:', name);
      const response = await fetch('http://localhost:5000/generate_logo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name }),
      });

      console.log('Logo response status:', response.status);
      
      if (response.ok) {
        const data = await response.json();
        console.log('Logo response data:', data);
        
        if (data.business_logo_url) {
          setLogoUrl(data.business_logo_url);
          console.log('Logo URL set successfully');
        } else {
          setError('No logo URL received from server');
        }
      } else {
        const errorData = await response.text();
        setError(`Failed to generate logo: ${response.status} - ${errorData}`);
      }
    } catch (err) {
      console.error('Logo generation error:', err);
      setError('Cannot connect to server');
    }
    setLoading(false);
  };

  const styles = {
    app: {
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #667eea 100%)',
      padding: '20px',
      fontFamily: 'Arial, sans-serif'
    },
    container: {
      maxWidth: '800px',
      margin: '0 auto',
      color: 'white',
      textAlign: 'center'
    },
    form: {
      background: 'rgba(255,255,255,0.15)',
      padding: '30px',
      borderRadius: '15px',
      marginBottom: '30px',
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(255,255,255,0.2)'
    },
    input: {
      width: '300px',
      padding: '12px',
      margin: '10px',
      border: 'none',
      borderRadius: '8px',
      fontSize: '16px'
    },
    button: {
      padding: '15px 30px',
      background: 'linear-gradient(45deg, #ff6b6b, #ee5a24)',
      color: 'white',
      border: 'none',
      borderRadius: '25px',
      fontSize: '16px',
      cursor: 'pointer',
      margin: '10px',
      fontWeight: 'bold',
      boxShadow: '0 4px 15px rgba(0,0,0,0.2)',
      transition: 'all 0.3s ease'
    },
    nameCard: {
      background: 'rgba(255,255,255,0.95)',
      color: '#333',
      padding: '15px 20px',
      margin: '10px',
      borderRadius: '12px',
      cursor: 'pointer',
      display: 'inline-block',
      fontWeight: 'bold',
      boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
      transition: 'all 0.3s ease',
      minWidth: '200px'
    },
    logo: {
      maxWidth: '200px',
      background: 'white',
      padding: '20px',
      borderRadius: '10px',
      margin: '20px'
    }
  };

  return (
    <div style={styles.app}>
      <div style={styles.container}>
        <h1>🚀 Business Name & Logo Generator</h1>
        <p>Generate creative business names and logos</p>

        <form onSubmit={generateNames} style={styles.form}>
          <div>
            <input
              type="text"
              value={idea}
              onChange={(e) => setIdea(e.target.value)}
              placeholder="Business Idea (e.g., coffee shop)"
              style={styles.input}
              required
            />
          </div>
          <div>
            <input
              type="text"
              value={theme}
              onChange={(e) => setTheme(e.target.value)}
              placeholder="Theme (e.g., modern, vintage)"
              style={styles.input}
              required
            />
          </div>
          <button type="submit" disabled={loading} style={styles.button}>
            {loading ? '🔄 Generating...' : '🎯 Generate Names'}
          </button>
        </form>

        {error && (
          <div style={{color: '#ff6b6b', background: 'rgba(255,255,255,0.1)', padding: '15px', borderRadius: '8px', margin: '20px'}}>
            ⚠️ {error}
          </div>
        )}

        {names.length > 0 && (
          <div>
            <h2>🎉 Generated Names (Click to create logo):</h2>
            <div>
              {names.map((name, index) => (
                <div
                  key={index}
                  style={styles.nameCard}
                  onClick={() => generateLogo(name)}
                >
                  {name}
                </div>
              ))}
            </div>
          </div>
        )}

        {selectedName && logoUrl && (
          <div>
            <h2>🎨 Logo for "{selectedName}"</h2>
            <img src={logoUrl} alt={`${selectedName} logo`} style={styles.logo} />
            <br />
            <a 
              href={logoUrl} 
              download={`${selectedName.replace(/\s+/g, '_')}_logo.svg`}
              style={{...styles.button, textDecoration: 'none', display: 'inline-block'}}
            >
              💾 Download Logo
            </a>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
