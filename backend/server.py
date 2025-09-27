import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import base64
import logging
import requests
import urllib.parse
import json
import re

# Optional: load .env for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional: Gemini API (configured if API key provided)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
# Use dynamic import to avoid static analyzer unresolved import warnings when package isn't installed
try:
    from importlib import import_module
    genai = import_module("google.generativeai")  # pip install google-generativeai
    GEMINI_AVAILABLE = True
except Exception:
    genai = None
    GEMINI_AVAILABLE = False

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature flags / provider selection via environment
USE_EXTERNAL_APIS = os.getenv("USE_EXTERNAL_APIS", "true").lower() == "true"
NAME_PROVIDER = os.getenv("NAME_PROVIDER", "datamuse")  # datamuse | ml | template
LOGO_PROVIDER = os.getenv("LOGO_PROVIDER", "dicebear")   # dicebear | ml | svg

# Initialize ML Pipeline
try:
    from ml_pipeline.simple_models import SimplePipeline
    ml_pipeline = SimplePipeline()
    
    # Try to load existing model or train quickly
    if not ml_pipeline.load_pipeline():
        logger.info("Training lightweight ML model (this will be quick)...")
        ml_pipeline.train_pipeline()
    
    logger.info("✅ Lightweight ML Pipeline initialized successfully")
    USE_ML_PIPELINE = True
except Exception as e:
    logger.warning(f"⚠️  Failed to initialize ML Pipeline: {e}. Falling back to template-based generation.")
    USE_ML_PIPELINE = False


# -----------------------------
# External Providers (No API key)
# -----------------------------

def generate_names_datamuse(idea: str, theme: str, max_results: int = 12):
    """Generate business names using the free Datamuse API (no key required).
    Strategy: fetch adjectives related to theme and nouns related to idea, then combine.
    """
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "BusinessGenerator/1.0"})

        # Get adjectives for the theme (rel_jjb: adjectives commonly used to describe the noun)
        adj_url = "https://api.datamuse.com/words"
        adj_params = {"rel_jjb": theme, "max": 20}
        adj_resp = session.get(adj_url, params=adj_params, timeout=8)
        adj_words = [w.get("word", "") for w in adj_resp.json() if w.get("word")]

        # Get nouns similar in meaning to idea (ml: means like)
        noun_params = {"ml": idea, "max": 20}
        noun_resp = session.get(adj_url, params=noun_params, timeout=8)
        noun_words = [w.get("word", "") for w in noun_resp.json() if w.get("word")]

        # Fallback if API empty
        if not adj_words:
            adj_words = [theme]
        if not noun_words:
            noun_words = [idea]

        # Combine into candidate names
        def title(s: str):
            return s.replace("-", " ").replace("_", " ").title()

        candidates = set()
        for a in adj_words[:10]:
            for n in noun_words[:10]:
                a_t, n_t = title(a), title(n)
                forms = [
                    f"{a_t} {n_t}",
                    f"The {n_t} Co.",
                    f"{n_t} & {a_t}",
                    f"{a_t} {title(idea)}",
                    f"{a_t} {title(theme)} {title(idea)}",
                ]
                for f in forms:
                    if 4 <= len(f) <= 28:
                        candidates.add(f)

        # Deterministic shuffle for repeatability per input
        seed = hash((idea.lower(), theme.lower())) % (2**32)
        rng = random.Random(seed)
        candidates = list(candidates)
        rng.shuffle(candidates)
        return candidates[:max_results] if candidates else generate_template_names(idea, theme)

    except Exception as e:
        logger.warning(f"Datamuse generation failed: {e}")
        return generate_template_names(idea, theme)


def dicebear_logo_url(name: str) -> str:
    """Generate a DiceBear SVG avatar URL seeded by the business name (no key required).
    Uses the 'shapes' collection for clean, brand-like logos.
    Docs: https://www.dicebear.com/styles/shapes
    """
    seed = urllib.parse.quote(name)
    # You can tweak options: backgroundColor, radius, randomizeIds, etc.
    # Keep size moderate for UI. The service returns SVG.
    return (
        f"https://api.dicebear.com/7.x/shapes/svg?seed={seed}"
        f"&radius=20&backgroundColor=b6e3f4,c0aede,d1d4f9&randomizeIds=true"
    )


def _parse_names_from_text(text: str):
    """Try to parse JSON {"names":[...]} first; fallback to line extraction."""
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            obj = json.loads(text[start:end + 1])
            names = obj.get('names') if isinstance(obj, dict) else None
            if isinstance(names, list):
                return [str(n).strip() for n in names if str(n).strip()]
    except Exception:
        pass
    # Fallback: split lines, remove bullets and numbering
    lines = [l.strip('-•* ').strip() for l in text.splitlines()]
    names = [l for l in lines if 3 <= len(l) <= 40]
    # Deduplicate preserving order
    seen, result = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            result.append(n)
    return result[:12]


def _camel_case_join(words: list[str]) -> str:
    return ''.join(w[:1].upper() + w[1:] for w in words if w)


def enforce_constraints_on_names(user_message: str, names: list[str]) -> list[str]:
    """Apply simple constraints inferred from the user message (e.g., one word)."""
    lower = (user_message or '').lower()
    one_word = any(k in lower for k in ["one word", "one-word", "single word", "single-word", "1 word", "1-word"])

    cleaned: list[str] = []
    for n in names:
        nn = n.strip()
        if one_word:
            # Remove non-alphanumeric and concatenate words in CamelCase
            tokens = re.findall(r"[a-zA-Z0-9]+", nn)
            nn = _camel_case_join(tokens)
        cleaned.append(nn)

    # Deduplicate while preserving order
    seen = set()
    out = []
    for n in cleaned:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def generate_names_gemini_rest(idea: str, theme: str, user_message: str, history: list, api_key: str):
    """Fallback to Gemini REST API if SDK path fails."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-1.5-flash:generateContent?key=" + api_key
    )
    sys_prompt = (
        "You are BrandBot, an expert business naming assistant. "
        "Given a business idea and theme, propose catchy, brandable names. "
        "Keep names concise (1-3 words), avoid special characters, and ensure a unique vibe. "
        "After a brief helpful explanation, output a single JSON object on its own line exactly as: "
        '{"names":["Name One","Name Two", ...]}'
    )
    contents = [
        {"role": "user", "parts": [{"text": sys_prompt}]},
        {"role": "user", "parts": [{"text": f"Context => Idea: {idea}\nTheme: {theme}"}]},
    ]
    for h in history[-6:]:
        role = "user" if h.get("role") == "user" else "model"
        contents.append({"role": role, "parts": [{"text": str(h.get("content", ""))}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    payload = {"contents": contents}
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    data = r.json()
    text = ""
    try:
        parts = data["candidates"][0]["content"]["parts"]
        # Concatenate text parts if multiple
        text = "\n".join(p.get("text", "") for p in parts)
    except Exception:
        text = ""
    names = _parse_names_from_text(text)
    return text, names


def generate_names_gemini(idea: str, theme: str, user_message: str, history: list, api_key: str | None = None):
    """Generate business names using Gemini if API key and library are available.
    history: list of {role: 'user'|'assistant', content: '...'}
    Returns (assistant_text, names_list)
    """
    # Use provided api_key or read from environment dynamically
    api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or not GEMINI_AVAILABLE:
        raise RuntimeError("Gemini API not configured or library missing")

    genai.configure(api_key=api_key)
    # Provide a clear system instruction and use chat for better multi-turn handling
    sys_prompt = (
        "You are BrandBot, an expert business naming assistant. "
        "Given a business idea and theme, propose catchy, brandable names. "
        "Keep names concise (1-3 words), avoid special characters, and ensure a unique vibe. "
        "After a brief helpful explanation, output a single JSON object on its own line exactly as: "
        '{"names":["Name One","Name Two", ...]}'
    )

    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        system_instruction=sys_prompt,
    )

    # Convert history to the expected format
    chat_history = []
    for h in history[-6:]:
        role = 'user' if h.get('role') == 'user' else 'model'
        content = str(h.get('content', ''))
        chat_history.append({"role": role, "parts": [content]})

    chat = model.start_chat(history=chat_history)
    user_block = f"Context => Idea: {idea}\nTheme: {theme}\nUser: {user_message}"
    resp = chat.send_message(user_block)

    # Extract text robustly
    text = getattr(resp, 'text', None)
    if not text and getattr(resp, 'candidates', None):
        try:
            text = resp.candidates[0].content.parts[0].text
        except Exception:
            text = ""
    text = text or ""
    names = _parse_names_from_text(text)
    return text, names

def generate_svg_logo(idea):
    """Generates creative themed SVG logos based on business type."""
    idea_lower = idea.lower()
    
    # Choose a random, modern background color
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FED766', '#2AB7CA', '#F4A261', '#E76F51', '#2A9D8F', '#E9C46A']
    bg_color = random.choice(colors)
    
    # Determine logo type based on keywords
    logo_content = ""
    
    if any(word in idea_lower for word in ['tea', 'chai', 'brew', 'leaf']):
        # Tea cup logo
        logo_content = f"""
        <g transform="translate(128,128)">
          <!-- Tea cup -->
          <path d="M-40,-20 L40,-20 L35,40 L-35,40 Z" fill="white" stroke="none"/>
          <ellipse cx="0" cy="-20" rx="40" ry="8" fill="white"/>
          <!-- Handle -->
          <path d="M40,-10 Q60,-10 60,10 Q60,30 40,30" fill="none" stroke="white" stroke-width="6"/>
          <!-- Steam -->
          <path d="M-20,-35 Q-15,-50 -10,-35 Q-5,-50 0,-35" fill="none" stroke="white" stroke-width="3"/>
          <path d="M10,-35 Q15,-50 20,-35 Q25,-50 30,-35" fill="none" stroke="white" stroke-width="3"/>
        </g>"""
    
    elif any(word in idea_lower for word in ['food', 'restaurant', 'kitchen', 'cook', 'chef', 'puff', 'bakery', 'bread']):
        # Chef hat or food logo
        logo_content = f"""
        <g transform="translate(128,128)">
          <!-- Chef hat -->
          <ellipse cx="0" cy="10" rx="50" ry="15" fill="white"/>
          <path d="M-50,10 Q-50,-30 -30,-40 Q-10,-50 10,-40 Q30,-50 50,-40 Q50,-30 50,10" fill="white"/>
          <!-- Puff details -->
          <circle cx="-25" cy="-25" r="8" fill="white" opacity="0.8"/>
          <circle cx="0" cy="-35" r="10" fill="white" opacity="0.8"/>
          <circle cx="25" cy="-25" r="8" fill="white" opacity="0.8"/>
        </g>"""
    
    elif any(word in idea_lower for word in ['tech', 'digital', 'software', 'app', 'web', 'code']):
        # Tech/digital logo
        logo_content = f"""
        <g transform="translate(128,128)">
          <!-- Circuit pattern -->
          <rect x="-40" y="-40" width="80" height="80" fill="none" stroke="white" stroke-width="4" rx="8"/>
          <circle cx="-20" cy="-20" r="6" fill="white"/>
          <circle cx="20" cy="-20" r="6" fill="white"/>
          <circle cx="-20" cy="20" r="6" fill="white"/>
          <circle cx="20" cy="20" r="6" fill="white"/>
          <path d="M-20,-20 L20,-20 M-20,20 L20,20 M-20,-20 L-20,20 M20,-20 L20,20" stroke="white" stroke-width="3"/>
        </g>"""
    
    elif any(word in idea_lower for word in ['health', 'medical', 'care', 'wellness', 'fit']):
        # Health/medical logo
        logo_content = f"""
        <g transform="translate(128,128)">
          <!-- Medical cross -->
          <rect x="-10" y="-40" width="20" height="80" fill="white" rx="4"/>
          <rect x="-40" y="-10" width="80" height="20" fill="white" rx="4"/>
          <!-- Heart shape -->
          <path d="M0,25 C-20,5 -30,-10 -15,-25 C0,-30 0,-30 15,-25 C30,-10 20,5 0,25" fill="white" opacity="0.7"/>
        </g>"""
    
    elif any(word in idea_lower for word in ['art', 'design', 'creative', 'studio', 'paint']):
        # Art/creative logo
        logo_content = f"""
        <g transform="translate(128,128)">
          <!-- Palette -->
          <ellipse cx="0" cy="0" rx="45" ry="35" fill="white"/>
          <circle cx="15" cy="5" r="12" fill="none"/>
          <!-- Paint brush -->
          <path d="M25,-25 L40,-40 Q45,-45 50,-40 L35,-25 Z" fill="white"/>
          <rect x="20" y="-30" width="4" height="15" fill="white"/>
          <!-- Color dots -->
          <circle cx="-20" cy="-10" r="4" fill="{bg_color}" opacity="0.7"/>
          <circle cx="-10" cy="15" r="4" fill="#FF6B6B" opacity="0.7"/>
          <circle cx="10" cy="-15" r="4" fill="#4ECDC4" opacity="0.7"/>
        </g>"""
    
    elif any(word in idea_lower for word in ['shop', 'store', 'market', 'retail', 'buy', 'sell']):
        # Shopping/retail logo
        logo_content = f"""
        <g transform="translate(128,128)">
          <!-- Shopping bag -->
          <path d="M-30,0 L30,0 L25,40 L-25,40 Z" fill="white" rx="4"/>
          <path d="M-20,0 Q-20,-20 0,-20 Q20,-20 20,0" fill="none" stroke="white" stroke-width="4"/>
          <!-- Store front -->
          <rect x="-35" y="-40" width="70" height="35" fill="white" opacity="0.8"/>
          <rect x="-25" y="-30" width="15" height="20" fill="{bg_color}"/>
          <rect x="10" y="-30" width="15" height="20" fill="{bg_color}"/>
        </g>"""
    
    elif any(word in idea_lower for word in ['finance', 'money', 'bank', 'invest', 'pay']):
        # Finance logo
        logo_content = f"""
        <g transform="translate(128,128)">
          <!-- Dollar sign -->
          <path d="M0,-40 L0,40 M-20,-20 Q-20,-30 -10,-30 Q10,-30 10,-20 Q10,-10 -10,-10 Q-30,-10 -30,0 Q-30,10 -20,10 Q0,10 0,20 Q0,30 10,30 Q30,30 30,20" 
                fill="none" stroke="white" stroke-width="6"/>
          <!-- Coins -->
          <circle cx="-25" cy="25" r="8" fill="white" opacity="0.7"/>
          <circle cx="25" cy="25" r="8" fill="white" opacity="0.7"/>
        </g>"""
    
    else:
        # Generic modern geometric logo
        logo_content = f"""
        <g transform="translate(128,128)">
          <!-- Modern geometric design -->
          <polygon points="-40,30 0,-40 40,30" fill="white" opacity="0.9"/>
          <circle cx="0" cy="10" r="20" fill="white" opacity="0.7"/>
          <rect x="-15" y="-5" width="30" height="30" fill="{bg_color}" opacity="0.8" rx="4"/>
        </g>"""

    # Create SVG string with the themed logo
    svg_string = f"""
    <svg width="256" height="256" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="bgGrad" cx="50%" cy="50%" r="50%">
          <stop offset="0%" style="stop-color:{bg_color};stop-opacity:1" />
          <stop offset="100%" style="stop-color:{bg_color};stop-opacity:0.8" />
        </radialGradient>
      </defs>
      <rect width="100%" height="100%" fill="url(#bgGrad)" />
      {logo_content}
    </svg>
    """

    # Base64 encode the SVG to embed it directly in the <img> tag
    b64_svg = base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64_svg}"

@app.route('/generate_business_names', methods=['POST'])
def generate_business_names():
    data = request.get_json()
    idea = data.get('idea')
    theme = data.get('theme')

    if not idea or not theme:
        return jsonify({"error": "Business idea and theme are required."}), 400

    try:
        business_names = None
        # Prefer external free provider if enabled
        if USE_EXTERNAL_APIS and NAME_PROVIDER == "datamuse":
            business_names = generate_names_datamuse(idea, theme)
        # Then try ML pipeline if available
        if not business_names and USE_ML_PIPELINE and NAME_PROVIDER in ("ml", "auto"):
            business_names = ml_pipeline.generate_business_names(idea, theme)
        # Finally fallback to templates
        if not business_names:
            business_names = generate_template_names(idea, theme)

        return jsonify({
            "business_names": business_names
        })

    except Exception as e:
        logger.error(f"Error generating business names: {e}")
        # Fallback to template generation
        business_names = generate_template_names(idea, theme)
        return jsonify({
            "business_names": business_names
        })

def generate_template_names(idea, theme):
    """Fallback template-based name generation."""
    name_templates = [
        f"{theme.capitalize()} {idea.capitalize()}",
        f"The {idea.capitalize()} Co.",
        f"{idea.capitalize()} & {theme.capitalize()}",
        f"Innovative {idea.capitalize()} Solutions",
        f"{idea.capitalize()} Hub",
        f"Simply {idea.capitalize()}",
        f"The Art of {idea.capitalize()}",
        f"{theme.capitalize()} Sprouts"
    ]
    return random.sample(name_templates, 5)

@app.route('/generate_logo', methods=['POST'])
def generate_logo():
    data = request.get_json()
    name = data.get('name')

    if not name:
        return jsonify({"error": "A business name is required."}), 400

    try:
        logo_url = None
        # Prefer external free DiceBear provider if enabled
        if USE_EXTERNAL_APIS and LOGO_PROVIDER == "dicebear":
            logo_url = dicebear_logo_url(name)
        # Then try ML pipeline
        if not logo_url and USE_ML_PIPELINE and LOGO_PROVIDER in ("ml", "auto"):
            logo_url = ml_pipeline.logo_generator.generate_logo(name)
        # Finally fallback to simple inline SVG
        if not logo_url:
            logo_url = generate_svg_logo(name)

        return jsonify({
            "business_logo_url": logo_url
        })

    except Exception as e:
        logger.error(f"Error generating logo: {e}")
        # Fallback to simple generation
        logo_url = generate_svg_logo(name)
        return jsonify({
            "business_logo_url": logo_url
        })


@app.route('/chat_names', methods=['POST'])
def chat_names():
    """Chat endpoint that uses Gemini to propose business names.
    Payload: { idea, theme, message, history: [{role, content}] }
    """
    data = request.get_json() or {}
    idea = (data.get('idea') or '').strip()
    theme = (data.get('theme') or '').strip()
    message = (data.get('message') or '').strip()
    history = data.get('history') or []

    if not idea or not theme:
        return jsonify({"error": "Both idea and theme are required"}), 400
    if not message:
        return jsonify({"error": "A message is required"}), 400

    # Ensure latest .env is loaded so newly added keys are picked up without full restart
    try:
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv()
    except Exception:
        pass

    current_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    # If Gemini isn't configured/available, gracefully fall back to free provider (Datamuse)
    if not current_key or not GEMINI_AVAILABLE:
        try:
            names = generate_names_datamuse(idea, theme, max_results=8)
        except Exception:
            names = generate_template_names(idea, theme)
        assistant_text = (
            "Gemini chat is not configured on the server, so I'm using a free provider "
            "to suggest names. To enable Gemini, set GEMINI_API_KEY on the server."
        )
        return jsonify({
            "assistant": assistant_text,
            "names": enforce_constraints_on_names(message, names)
        })

    try:
        assistant_text, names = generate_names_gemini(idea, theme, message, history, api_key=current_key)
        # If Gemini returns no names, optionally fall back to Datamuse combo based on message keywords
        if not names:
            names = generate_names_datamuse(idea, theme, max_results=8)
        return jsonify({
            "assistant": assistant_text,
            "names": enforce_constraints_on_names(message, names)
        })
    except Exception as e:
        logger.warning(f"Gemini SDK chat failed: {e}. Trying REST fallback...")
        try:
            assistant_text, names = generate_names_gemini_rest(idea, theme, message, history, api_key=current_key)
            if not names:
                names = generate_names_datamuse(idea, theme, max_results=8)
            return jsonify({
                "assistant": assistant_text or "Here are some ideas.",
                "names": enforce_constraints_on_names(message, names)
            })
        except Exception as e2:
            logger.error(f"Gemini REST fallback failed: {e2}. Using free provider.")
            try:
                names = generate_names_datamuse(idea, theme, max_results=8)
            except Exception:
                names = generate_template_names(idea, theme)
            assistant_text = (
                "I couldn't reach Gemini just now, so here are suggestions from a free provider. "
                "You can retry or adjust your prompt."
            )
            return jsonify({
                "assistant": assistant_text,
                "names": enforce_constraints_on_names(message, names)
            })

@app.route('/train_pipeline', methods=['POST'])
def train_pipeline():
    """Endpoint to trigger ML pipeline training."""
    try:
        if not USE_ML_PIPELINE:
            return jsonify({"error": "ML Pipeline not available"}), 500
        
        logger.info("Starting ML pipeline training...")
        ml_pipeline.train_pipeline()
        
        return jsonify({
            "message": "Pipeline training completed successfully",
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error training pipeline: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/evaluate_pipeline', methods=['GET'])
def evaluate_pipeline():
    """Endpoint to evaluate the ML pipeline."""
    try:
        if not USE_ML_PIPELINE:
            return jsonify({"error": "ML Pipeline not available"}), 500
        
        results = ml_pipeline.evaluate_pipeline()
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error evaluating pipeline: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "healthy",
        "server": "Business Generator API",
        "timestamp": str(Path(__file__).stat().st_mtime)
    })

@app.route('/status', methods=['GET'])
def status():
    """Expose key feature availability (e.g., Gemini)."""
    # Re-read env to reflect any recent .env changes without full restart
    try:
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv()
    except Exception:
        pass
    current_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return jsonify({
        "gemini_configured": bool(current_key),
        "gemini_available": bool(current_key and GEMINI_AVAILABLE),
        "external_apis": USE_EXTERNAL_APIS,
        "name_provider": NAME_PROVIDER,
        "logo_provider": LOGO_PROVIDER
    })

@app.route('/pipeline_status', methods=['GET'])
def pipeline_status():
    """Get the status of the ML pipeline."""
    return jsonify({
        "ml_pipeline_enabled": USE_ML_PIPELINE,
        "model_available": USE_ML_PIPELINE and ml_pipeline.name_generator.model is not None,
        "status": "ready" if USE_ML_PIPELINE else "fallback_mode"
    })

if __name__ == '__main__':
    logger.info("Starting Business Generator API Server...")
    logger.info(f"ML Pipeline enabled: {USE_ML_PIPELINE}")
    logger.info(f"External APIs enabled: {USE_EXTERNAL_APIS} (names={NAME_PROVIDER}, logos={LOGO_PROVIDER})")
    app.run(port=5000, debug=True)
