import logging
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from .config import MODEL_DIR, LOGGING_CONFIG
from .models import BusinessNameTransformer, create_model
from .data_pipeline import prepare_data_pipeline
import logging.config

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class BusinessNameGenerator:
    """Production inference class for business name generation."""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        
        if model_path:
            self.load_model(model_path)
        else:
            # Try to load the best model
            best_model_path = MODEL_DIR / 'best_business_name_model.pt'
            if best_model_path.exists():
                self.load_model(str(best_model_path))
            else:
                logger.warning("No trained model found. Please train a model first.")
    
    def load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.vocab_to_idx = checkpoint['vocab_to_idx']
            self.idx_to_vocab = checkpoint['idx_to_vocab']
            
            # Create model with same architecture
            self.model = create_model(len(self.vocab_to_idx))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_names(self, industry: str, theme: str, num_names: int = 5) -> List[str]:
        """Generate business names based on industry and theme."""
        if self.model is None:
            logger.error("No model loaded. Cannot generate names.")
            return []
        
        try:
            # Prepare input
            input_text = f"{industry.lower()} {theme.lower()}"
            input_tokens = input_text.split()
            
            # Convert to tensor
            input_ids = [self.vocab_to_idx.get(token, self.vocab_to_idx.get('<UNK>', 3)) 
                        for token in input_tokens]
            
            # Pad to model's expected length
            max_length = 50
            if len(input_ids) < max_length:
                input_ids.extend([self.vocab_to_idx['<PAD>']] * (max_length - len(input_ids)))
            else:
                input_ids = input_ids[:max_length]
            
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            
            # Generate multiple names
            generated_names = []
            
            for _ in range(num_names):
                try:
                    generated_text = self.model.generate(
                        input_tensor, self.vocab_to_idx, self.idx_to_vocab
                    )
                    
                    # Clean and format the generated text
                    cleaned_name = self._clean_generated_name(generated_text)
                    if cleaned_name and cleaned_name not in generated_names:
                        generated_names.append(cleaned_name)
                        
                except Exception as e:
                    logger.warning(f"Error generating name: {e}")
                    continue
            
            # If we don't have enough unique names, add some fallback names
            if len(generated_names) < num_names:
                fallback_names = self._generate_fallback_names(industry, theme, 
                                                              num_names - len(generated_names))
                generated_names.extend(fallback_names)
            
            return generated_names[:num_names]
            
        except Exception as e:
            logger.error(f"Error in name generation: {e}")
            # Return fallback names in case of error
            return self._generate_fallback_names(industry, theme, num_names)
    
    def _clean_generated_name(self, text: str) -> str:
        """Clean and format generated business name."""
        if not text:
            return ""
        
        # Remove extra spaces and clean
        cleaned = ' '.join(text.split())
        
        # Capitalize each word
        words = cleaned.split()
        capitalized_words = []
        
        for word in words:
            if word.lower() in ['and', 'or', 'of', 'the', 'a', 'an', 'in', 'on', 'at']:
                capitalized_words.append(word.lower())
            else:
                capitalized_words.append(word.capitalize())
        
        return ' '.join(capitalized_words)
    
    def _generate_fallback_names(self, industry: str, theme: str, num_names: int) -> List[str]:
        """Generate fallback names using templates."""
        templates = [
            f"{theme.capitalize()} {industry.capitalize()} Solutions",
            f"The {theme.capitalize()} Company",
            f"{industry.capitalize()} {theme.capitalize()} Hub",
            f"Innovative {theme.capitalize()} Systems",
            f"{theme.capitalize()} & Associates",
            f"Premier {industry.capitalize()} Services",
            f"{theme.capitalize()} Dynamics",
            f"Future {theme.capitalize()} Technologies",
            f"{industry.capitalize()} {theme.capitalize()} Partners",
            f"Elite {theme.capitalize()} Group"
        ]
        
        # Shuffle and return requested number
        import random
        random.shuffle(templates)
        return templates[:num_names]

class LogoGenerator:
    """Simple logo generation for the pipeline."""
    
    def __init__(self):
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FED766', '#2AB7CA', 
            '#F4A261', '#E76F51', '#2A9D8F', '#E9C46A', '#FF8E53'
        ]
    
    def generate_logo(self, business_name: str) -> str:
        """Generate a simple SVG logo based on business name."""
        import random
        import base64
        
        # Extract initials
        words = business_name.split()
        initials = ""
        
        if len(words) >= 2:
            initials = words[0][0] + words[1][0]
        elif len(words) == 1 and len(words[0]) >= 2:
            initials = words[0][:2]
        elif len(words) == 1:
            initials = words[0][0]
        else:
            initials = "BG"
        
        initials = initials.upper()
        
        # Choose colors
        bg_color = random.choice(self.colors)
        text_color = "#FFFFFF"
        
        # Create more sophisticated SVG
        svg_content = f"""
        <svg width="256" height="256" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:{bg_color};stop-opacity:1" />
                    <stop offset="100%" style="stop-color:{self._darken_color(bg_color)};stop-opacity:1" />
                </linearGradient>
            </defs>
            <rect width="100%" height="100%" fill="url(#grad1)" rx="20" ry="20"/>
            <circle cx="128" cy="128" r="100" fill="none" stroke="{text_color}" stroke-width="2" opacity="0.3"/>
            <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" 
                  font-family="Arial, sans-serif" font-size="72" font-weight="bold" 
                  fill="{text_color}">
                {initials}
            </text>
        </svg>
        """
        
        # Encode to base64
        b64_svg = base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
        return f"data:image/svg+xml;base64,{b64_svg}"
    
    def _darken_color(self, hex_color: str) -> str:
        """Darken a hex color for gradient effect."""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darkened_rgb = tuple(max(0, int(c * 0.7)) for c in rgb)
        return f"#{darkened_rgb[0]:02x}{darkened_rgb[1]:02x}{darkened_rgb[2]:02x}"

class MLPipeline:
    """Main ML Pipeline orchestrator."""
    
    def __init__(self):
        self.name_generator = BusinessNameGenerator()
        self.logo_generator = LogoGenerator()
        logger.info("ML Pipeline initialized")
    
    def generate_business_suggestions(self, industry: str, theme: str, 
                                    num_suggestions: int = 5) -> List[Dict]:
        """Generate complete business suggestions with names and logos."""
        try:
            # Generate business names
            business_names = self.name_generator.generate_names(industry, theme, num_suggestions)
            
            # Generate suggestions with logos
            suggestions = []
            for name in business_names:
                logo_url = self.logo_generator.generate_logo(name)
                suggestions.append({
                    'name': name,
                    'logo_url': logo_url,
                    'industry': industry,
                    'theme': theme
                })
            
            logger.info(f"Generated {len(suggestions)} business suggestions")
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating business suggestions: {e}")
            return []
    
    def train_pipeline(self):
        """Train the ML models in the pipeline."""
        logger.info("Starting ML pipeline training...")
        
        try:
            # Prepare data
            train_loader, val_loader, vocab_to_idx, idx_to_vocab = prepare_data_pipeline()
            
            # Train business name model
            from .models import train_business_name_model
            trainer = train_business_name_model(train_loader, val_loader, vocab_to_idx, idx_to_vocab)
            
            # Reload the trained model
            self.name_generator = BusinessNameGenerator()
            
            logger.info("ML pipeline training completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during pipeline training: {e}")
            raise
    
    def evaluate_pipeline(self) -> Dict:
        """Evaluate the pipeline performance."""
        try:
            # Test with sample inputs
            test_cases = [
                ("technology", "software"),
                ("healthcare", "wellness"),
                ("finance", "investment"),
                ("retail", "fashion"),
                ("food", "restaurant")
            ]
            
            results = {"test_results": []}
            
            for industry, theme in test_cases:
                suggestions = self.generate_business_suggestions(industry, theme, 3)
                results["test_results"].append({
                    "input": {"industry": industry, "theme": theme},
                    "output": suggestions,
                    "success": len(suggestions) > 0
                })
            
            # Calculate success rate
            success_count = sum(1 for result in results["test_results"] if result["success"])
            results["success_rate"] = success_count / len(test_cases)
            
            logger.info(f"Pipeline evaluation completed. Success rate: {results['success_rate']:.2%}")
            return results
            
        except Exception as e:
            logger.error(f"Error during pipeline evaluation: {e}")
            return {"error": str(e)}
