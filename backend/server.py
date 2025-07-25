import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import base64
import logging

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        if USE_ML_PIPELINE:
            # Use lightweight ML Pipeline for generation
            business_names = ml_pipeline.generate_business_names(idea, theme)
        else:
            # Fallback to template-based generation
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
        if USE_ML_PIPELINE:
            # Use ML Pipeline for logo generation
            logo_url = ml_pipeline.logo_generator.generate_logo(name)
        else:
            # Fallback to simple SVG generation
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
    app.run(port=5000, debug=True)
