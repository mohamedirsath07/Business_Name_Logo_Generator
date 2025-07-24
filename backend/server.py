import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import random
import base64

app = Flask(__name__)
CORS(app)

# Set the OpenAI API key from environment variable
# Create a .env file in the backend directory and add your API key there
openai.api_key = os.getenv('OPENAI_API_KEY', '')

def generate_svg_logo(idea):
    """Generates a simple, professional SVG logo with the business initials."""
    words = idea.split()
    initials = ""
    if len(words) >= 2:
        initials = words[0][0] + words[1][0]
    elif len(words) == 1 and len(words[0]) >= 2:
        initials = words[0][:2]
    elif len(words) == 1:
        initials = words[0][0]
    else:
        initials = "B"  # Fallback

    initials = initials.upper()

    # Choose a random, modern background color
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FED766', '#2AB7CA', '#F4A261', '#E76F51', '#2A9D8F', '#E9C46A']
    bg_color = random.choice(colors)

    # Create SVG string for a clean, modern logo
    svg_string = f"""
    <svg width="256" height="256" xmlns="http://www.w3.org/2000/svg">
      <rect width="100%" height="100%" fill="{bg_color}" />
      <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="Arial, sans-serif" font-size="96" font-weight="bold" fill="#FFFFFF">
        {initials}
      </text>
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
        # MOCK DATA GENERATION
        # Generate more creative placeholder business names
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
        cleaned_names = random.sample(name_templates, 5)

        return jsonify({
            "business_names": cleaned_names
        })

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected error occurred. Please check the server logs."}), 500

@app.route('/generate_logo', methods=['POST'])
def generate_logo():
    data = request.get_json()
    name = data.get('name')

    if not name:
        return jsonify({"error": "A business name is required."}), 400

    try:
        # Generate a professional SVG logo
        logo_url = generate_svg_logo(name)

        return jsonify({
            "business_logo_url": logo_url
        })

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected error occurred. Please check the server logs."}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
