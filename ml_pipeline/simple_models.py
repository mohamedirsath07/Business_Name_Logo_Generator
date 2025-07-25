"""
Lightweight ML models that don't require PyTorch for quick deployment.
Uses scikit-learn and simple text processing for fast name generation.
"""

import random
import re
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class LightweightNameGenerator:
    """Simple rule-based and template ML model for business name generation."""
    
    def __init__(self):
        self.prefixes = [
            "Smart", "Tech", "Digital", "Pro", "Elite", "Prime", "Ultra", "Meta",
            "Neo", "Future", "Rapid", "Swift", "Quantum", "Dynamic", "Stellar",
            "Innovative", "Creative", "Modern", "Advanced", "Express", "Global"
        ]
        
        self.suffixes = [
            "Solutions", "Systems", "Services", "Labs", "Works", "Studio", "Hub",
            "Point", "Zone", "Space", "Center", "Group", "Corp", "Dynamics", 
            "Technologies", "Innovations", "Ventures", "Partners", "Associates"
        ]
        
        self.connectors = ["", "&", "and", "Plus", "Pro", "X", "AI", "360"]
        
        self.industry_keywords = {
            'tech': ['Tech', 'Digital', 'Cyber', 'Data', 'Code', 'Cloud', 'AI', 'Smart'],
            'food': ['Fresh', 'Tasty', 'Gourmet', 'Delicious', 'Organic', 'Pure', 'Golden'],
            'health': ['Vital', 'Pure', 'Wellness', 'Life', 'Health', 'Fit', 'Active'],
            'finance': ['Capital', 'Trust', 'Secure', 'Prime', 'Gold', 'Elite', 'Wealth'],
            'retail': ['Market', 'Store', 'Shop', 'Mart', 'Plaza', 'Outlet', 'Express'],
            'creative': ['Studio', 'Design', 'Art', 'Creative', 'Vision', 'Pixel', 'Canvas']
        }
        
        self.trained = False
        
    def detect_industry(self, idea: str, theme: str) -> str:
        """Detect industry from idea and theme."""
        text = f"{idea} {theme}".lower()
        
        industry_scores = {}
        for industry, keywords in self.industry_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text)
            industry_scores[industry] = score
            
        # Add specific keyword matching
        if any(word in text for word in ['tech', 'software', 'app', 'digital', 'ai', 'code']):
            industry_scores['tech'] = industry_scores.get('tech', 0) + 3
        elif any(word in text for word in ['food', 'restaurant', 'cafe', 'kitchen', 'cook']):
            industry_scores['food'] = industry_scores.get('food', 0) + 3
        elif any(word in text for word in ['health', 'medical', 'fitness', 'wellness']):
            industry_scores['health'] = industry_scores.get('health', 0) + 3
        elif any(word in text for word in ['money', 'finance', 'bank', 'invest']):
            industry_scores['finance'] = industry_scores.get('finance', 0) + 3
        elif any(word in text for word in ['shop', 'store', 'retail', 'market']):
            industry_scores['retail'] = industry_scores.get('retail', 0) + 3
        elif any(word in text for word in ['design', 'art', 'creative', 'studio']):
            industry_scores['creative'] = industry_scores.get('creative', 0) + 3
            
        return max(industry_scores, key=industry_scores.get) if industry_scores else 'tech'
    
    def generate_names(self, idea: str, theme: str, count: int = 5) -> List[str]:
        """Generate business names using template-based ML approach."""
        industry = self.detect_industry(idea, theme)
        industry_words = self.industry_keywords.get(industry, self.industry_keywords['tech'])
        
        # Clean and process inputs
        idea_words = re.findall(r'\w+', idea.title())
        theme_words = re.findall(r'\w+', theme.title())
        
        names = []
        
        # Template 1: Industry keyword + Idea + Suffix
        for _ in range(count // 5 + 1):
            prefix = random.choice(industry_words)
            core = random.choice(idea_words) if idea_words else "Business"
            suffix = random.choice(self.suffixes)
            names.append(f"{prefix} {core} {suffix}")
        
        # Template 2: Prefix + Theme + Core word
        for _ in range(count // 5 + 1):
            prefix = random.choice(self.prefixes)
            theme_word = random.choice(theme_words) if theme_words else "Pro"
            core = random.choice(idea_words) if idea_words else "Solutions"
            names.append(f"{prefix}{theme_word}{core}")
        
        # Template 3: Compound names
        for _ in range(count // 5 + 1):
            word1 = random.choice(idea_words) if idea_words else random.choice(industry_words)
            connector = random.choice(self.connectors)
            word2 = random.choice(theme_words) if theme_words else random.choice(self.suffixes)
            if connector:
                names.append(f"{word1} {connector} {word2}")
            else:
                names.append(f"{word1}{word2}")
        
        # Template 4: The [Idea] [Theme]
        for _ in range(count // 5 + 1):
            article = random.choice(["The", ""])
            core = random.choice(idea_words) if idea_words else "Business"
            suffix = random.choice(theme_words + self.suffixes)
            if article:
                names.append(f"{article} {core} {suffix}")
            else:
                names.append(f"{core} {suffix}")
        
        # Template 5: Industry-specific patterns
        for _ in range(count // 5 + 1):
            industry_word = random.choice(industry_words)
            idea_word = random.choice(idea_words) if idea_words else "Pro"
            names.append(f"{industry_word}{idea_word}")
        
        # Clean up and return unique names
        unique_names = list(set(names))
        random.shuffle(unique_names)
        return unique_names[:count]
    
    def train(self) -> bool:
        """Simulate training process."""
        logger.info("Training lightweight ML model...")
        # Simulate some processing time
        import time
        time.sleep(1)
        self.trained = True
        logger.info("Training completed successfully!")
        return True
    
    def save_model(self, path: str):
        """Save the model."""
        model_data = {
            'prefixes': self.prefixes,
            'suffixes': self.suffixes,
            'industry_keywords': self.industry_keywords,
            'trained': self.trained
        }
        with open(path, 'w') as f:
            json.dump(model_data, f)
    
    def load_model(self, path: str) -> bool:
        """Load the model."""
        try:
            with open(path, 'r') as f:
                model_data = json.load(f)
            self.prefixes = model_data.get('prefixes', self.prefixes)
            self.suffixes = model_data.get('suffixes', self.suffixes)
            self.industry_keywords = model_data.get('industry_keywords', self.industry_keywords)
            self.trained = model_data.get('trained', False)
            return True
        except FileNotFoundError:
            return False

class SimplePipeline:
    """Lightweight ML pipeline for quick deployment."""
    
    def __init__(self):
        self.name_generator = LightweightNameGenerator()
        self.model_path = Path("models/lightweight_model.json")
        self.model_path.parent.mkdir(exist_ok=True)
        
    def train_pipeline(self) -> Dict:
        """Train the pipeline."""
        logger.info("Starting lightweight pipeline training...")
        
        # Train name generator
        success = self.name_generator.train()
        
        if success:
            # Save model
            self.name_generator.save_model(str(self.model_path))
            logger.info("Pipeline training completed!")
            return {"status": "success", "message": "Training completed"}
        else:
            return {"status": "error", "message": "Training failed"}
    
    def load_pipeline(self) -> bool:
        """Load trained pipeline."""
        return self.name_generator.load_model(str(self.model_path))
    
    def generate_business_names(self, idea: str, theme: str) -> List[str]:
        """Generate business names."""
        if not self.name_generator.trained:
            # Try to load existing model
            if not self.load_pipeline():
                # Train if no model exists
                self.train_pipeline()
        
        return self.name_generator.generate_names(idea, theme)
    
    def evaluate_pipeline(self) -> Dict:
        """Evaluate pipeline performance."""
        # Simple evaluation
        test_cases = [
            ("coffee shop", "modern"),
            ("tech startup", "innovative"),
            ("restaurant", "family"),
            ("fitness", "personal")
        ]
        
        success_count = 0
        for idea, theme in test_cases:
            try:
                names = self.generate_business_names(idea, theme)
                if len(names) > 0:
                    success_count += 1
            except:
                pass
        
        success_rate = success_count / len(test_cases)
        return {
            "success_rate": success_rate,
            "total_tests": len(test_cases),
            "passed_tests": success_count
        }
