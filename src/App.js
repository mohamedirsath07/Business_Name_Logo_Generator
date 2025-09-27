// Simple Business Name & Logo Generator
import React, { useState, useRef, useEffect } from 'react';
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
  // Removed connection test UI
  const [chatOpen, setChatOpen] = useState(false);
  const [chatHistory, setChatHistory] = useState([]); // {role: 'user'|'assistant', content}
  const [chatNames, setChatNames] = useState([]); // last suggested names
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const logoRef = useRef(null);
  const namesRef = useRef(null);
  const chatRef = useRef(null);

  // Scroll to names section when names arrive
  useEffect(() => {
    if (names && names.length > 0) {
      try { namesRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }); } catch {}
    }
  }, [names]);

  // Scroll to chat section when opened
  useEffect(() => {
    if (chatOpen) {
      // Defer to next paint to ensure DOM exists
      setTimeout(() => {
        try { chatRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }); } catch {}
      }, 0);
    }
  }, [chatOpen]);

  // (Removed testConnection)

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

  // --- Chat with Gemini for name suggestions ---
  const sendChat = async () => {
    if (!idea.trim() || !theme.trim()) {
      setError('Please fill business idea and theme before chatting');
      return;
    }
    if (!chatInput.trim()) return;

    const newHistory = [...chatHistory, { role: 'user', content: chatInput }];
    setChatHistory(newHistory);
    setChatLoading(true);
    try {
      const resp = await fetch('http://localhost:5000/chat_names', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          idea: idea.trim(),
          theme: theme.trim(),
          message: chatInput.trim(),
          history: newHistory,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || 'Chat request failed');
      }
      // Append assistant message
      setChatHistory((h) => [...h, { role: 'assistant', content: data.assistant || 'Here are some ideas!' }]);
      if (Array.isArray(data.names) && data.names.length) {
        // Merge suggestions into the names list (dedupe)
        const merged = Array.from(new Set([...(names || []), ...data.names]));
        setNames(merged);
        setChatNames(data.names);
      }
      setChatInput('');
    } catch (e) {
      console.error('Chat error', e);
      setError(String(e.message || e));
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="App-header">
        <h1>🚀 Business Name & Logo Generator</h1>
        <p>Enter your business idea and theme to generate creative names and logos</p>
        
        {/* Removed Test Server Connection button */}
        
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
          
          <div style={{ display: 'flex', gap: 12 }}>
            <button type="submit" disabled={loadingNames} className="generate-btn btn-light" style={{ flex: 1 }}>
            {loadingNames ? '🔄 Generating Names...' : 'Generate Business Names'}
            </button>
            <button
              type="button"
              className="generate-btn btn-danger"
              style={{ flex: 1 }}
              onClick={() => setChatOpen((o) => !o)}
            >
              {chatOpen ? 'Close Name Chat' : '💬 Open Name Chat'}
            </button>
          </div>
        </form>

        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}

        {names.length > 0 && (
          <div className="results-section" ref={namesRef}>
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

        {chatOpen && (
          <div className="results-section" ref={chatRef} style={{ textAlign: 'left' }}>
            <h2>💬 Name Chat Assistant</h2>
            <div style={{
              height: 220,
              overflowY: 'auto',
              padding: '12px 14px',
              borderRadius: 12,
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,255,255,0.16)'
            }}>
              {chatHistory.length === 0 && (
                <div style={{ opacity: 0.8 }}>
                  Tip: Tell the assistant about tone (playful, luxury), constraints (max 2 words, available domains), or industry terms to include/exclude.
                </div>
              )}
              {chatHistory.map((m, i) => (
                <div key={i} style={{
                  margin: '8px 0',
                  display: 'flex',
                  justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start'
                }}>
                  <div style={{
                    maxWidth: '80%',
                    whiteSpace: 'pre-wrap',
                    padding: '10px 12px',
                    borderRadius: 10,
                    background: m.role === 'user' ? 'rgba(52,211,153,0.2)' : 'rgba(255,255,255,0.08)',
                    border: '1px solid rgba(255,255,255,0.16)'
                  }}>
                    {m.content}
                  </div>
                </div>
              ))}
            </div>
            {chatNames.length > 0 && (
              <div style={{ marginTop: 12 }}>
                <div style={{ fontSize: 13, opacity: 0.8, marginBottom: 6 }}>Suggested names (tap to add logo-ready card):</div>
                <div className="names-grid">
                  {chatNames.map((n, idx) => (
                    <button
                      key={idx}
                      className="name-card"
                      onClick={() => {
                        if (!names.includes(n)) setNames([...names, n]);
                      }}
                    >{n}</button>
                  ))}
                </div>
              </div>
            )}
            <div className="chat-input-row">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Describe your preference, constraints, or tone..."
                className="chat-input"
              />
              <button type="button" className="generate-btn chat-send-btn" disabled={chatLoading} onClick={sendChat}>
                {chatLoading ? '✍️ Thinking...' : 'Send'}
              </button>
            </div>
          </div>
        )}

        {loadingLogo && (
          <div className="loading-message">
            🎨 Generating logo for "{selectedName}"...
          </div>
        )}

        {logoUrl && selectedName && (
          <div className="logo-section" ref={logoRef}>
            <h2>🎨 Logo for "{selectedName}"</h2>
            <div className="logo-container">
              <img
                src={logoUrl}
                alt={`${selectedName} logo`}
                className="logo-image"
                onLoad={() => {
                  try {
                    logoRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                  } catch (e) {
                    // no-op
                  }
                }}
              />
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