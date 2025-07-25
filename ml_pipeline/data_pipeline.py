import logging
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from typing import List, Dict, Tuple
import re
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from .config import DATA_DIR, DATA_SOURCES, LOGGING_CONFIG
import logging.config

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class BusinessNameDataCollector:
    """Collects and preprocesses business name data for training."""
    
    def __init__(self):
        self.raw_data_dir = DATA_DIR / "raw"
        self.processed_data_dir = DATA_DIR / "processed"
        self.raw_data_dir.mkdir(exist_ok=True)
        self.processed_data_dir.mkdir(exist_ok=True)
    
    def collect_business_names(self) -> pd.DataFrame:
        """Collect business names from various sources."""
        logger.info("Starting business name data collection...")
        
        all_names = []
        
        # Collect from online sources
        for source_url in DATA_SOURCES["business_names"]:
            try:
                logger.info(f"Collecting data from {source_url}")
                df = pd.read_csv(source_url)
                if 'name' in df.columns:
                    all_names.extend(df['name'].dropna().tolist())
                elif 'company_name' in df.columns:
                    all_names.extend(df['company_name'].dropna().tolist())
            except Exception as e:
                logger.warning(f"Failed to collect from {source_url}: {e}")
        
        # Add some synthetic business names for training
        synthetic_names = self._generate_synthetic_names()
        all_names.extend(synthetic_names)
        
        # Create DataFrame
        df = pd.DataFrame({'business_name': all_names})
        df = df.drop_duplicates().reset_index(drop=True)
        
        logger.info(f"Collected {len(df)} unique business names")
        return df
    
    def _generate_synthetic_names(self) -> List[str]:
        """Generate synthetic business names for training data augmentation."""
        prefixes = [
            "Innovative", "Creative", "Dynamic", "Modern", "Future", "Smart", "Digital",
            "Global", "Premier", "Elite", "Advanced", "Strategic", "Progressive", "Eco"
        ]
        
        core_words = [
            "Solutions", "Technologies", "Systems", "Services", "Consulting", "Labs",
            "Studio", "Works", "Group", "Partners", "Enterprises", "Innovations",
            "Dynamics", "Ventures", "Holdings", "Associates"
        ]
        
        industries = [
            "Tech", "Media", "Finance", "Health", "Education", "Retail", "Food",
            "Fashion", "Travel", "Energy", "Real Estate", "Automotive", "Sports"
        ]
        
        synthetic_names = []
        
        # Generate combination patterns
        for prefix in prefixes:
            for core in core_words:
                synthetic_names.append(f"{prefix} {core}")
        
        for industry in industries:
            for core in core_words:
                synthetic_names.append(f"{industry} {core}")
                synthetic_names.append(f"The {industry} {core}")
        
        # Add some creative combinations
        for i, industry in enumerate(industries):
            for j, prefix in enumerate(prefixes[:len(industries)]):
                if i != j:  # Avoid same index combinations
                    synthetic_names.append(f"{prefix} {industry} Hub")
                    synthetic_names.append(f"{industry} {prefix}")
        
        return synthetic_names[:1000]  # Limit to 1000 synthetic names
    
    def preprocess_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess business names for training."""
        logger.info("Preprocessing business names...")
        
        # Clean the names
        df['clean_name'] = df['business_name'].apply(self._clean_name)
        
        # Extract features
        df['length'] = df['clean_name'].str.len()
        df['word_count'] = df['clean_name'].str.split().str.len()
        df['has_llc'] = df['business_name'].str.contains(r'\b(LLC|Inc|Corp|Ltd)\b', case=False)
        df['starts_with_the'] = df['clean_name'].str.startswith('the ', case=False)
        
        # Categorize by industry (simple keyword matching)
        df['industry'] = df['clean_name'].apply(self._categorize_industry)
        
        # Filter out very short or very long names
        df = df[(df['length'] >= 3) & (df['length'] <= 50)].reset_index(drop=True)
        
        logger.info(f"After preprocessing: {len(df)} business names")
        return df
    
    def _clean_name(self, name: str) -> str:
        """Clean a business name."""
        if pd.isna(name):
            return ""
        
        # Convert to lowercase and strip
        clean = str(name).lower().strip()
        
        # Remove special characters but keep spaces and alphanumeric
        clean = re.sub(r'[^\w\s]', '', clean)
        
        # Remove extra whitespace
        clean = re.sub(r'\s+', ' ', clean)
        
        return clean
    
    def _categorize_industry(self, name: str) -> str:
        """Categorize business by industry based on keywords."""
        name_lower = name.lower()
        
        tech_keywords = ['tech', 'software', 'digital', 'data', 'cyber', 'ai', 'cloud']
        if any(keyword in name_lower for keyword in tech_keywords):
            return 'technology'
        
        health_keywords = ['health', 'medical', 'care', 'wellness', 'pharma']
        if any(keyword in name_lower for keyword in health_keywords):
            return 'healthcare'
        
        finance_keywords = ['finance', 'bank', 'invest', 'capital', 'fund']
        if any(keyword in name_lower for keyword in finance_keywords):
            return 'finance'
        
        retail_keywords = ['retail', 'shop', 'store', 'market', 'sales']
        if any(keyword in name_lower for keyword in retail_keywords):
            return 'retail'
        
        service_keywords = ['service', 'consulting', 'solution', 'support']
        if any(keyword in name_lower for keyword in service_keywords):
            return 'services'
        
        return 'general'
    
    def create_training_pairs(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Create input-output pairs for training."""
        logger.info("Creating training pairs...")
        
        training_pairs = []
        
        for _, row in df.iterrows():
            # Create pairs: (industry + theme) -> business_name
            industry = row['industry']
            name = row['clean_name']
            
            # Extract theme from the name (first meaningful word)
            words = name.split()
            if len(words) > 1:
                theme = words[0] if words[0] not in ['the', 'a', 'an'] else words[1]
            else:
                theme = words[0] if words else 'business'
            
            input_text = f"{industry} {theme}"
            output_text = name
            
            training_pairs.append((input_text, output_text))
        
        logger.info(f"Created {len(training_pairs)} training pairs")
        return training_pairs

class BusinessNameDataset(Dataset):
    """PyTorch Dataset for business name generation."""
    
    def __init__(self, training_pairs: List[Tuple[str, str]], vocab_to_idx: Dict[str, int], max_length: int = 50):
        self.training_pairs = training_pairs
        self.vocab_to_idx = vocab_to_idx
        self.max_length = max_length
        self.pad_token = vocab_to_idx.get('<PAD>', 0)
        self.start_token = vocab_to_idx.get('<START>', 1)
        self.end_token = vocab_to_idx.get('<END>', 2)
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        input_text, output_text = self.training_pairs[idx]
        
        # Tokenize input
        input_tokens = self._tokenize(input_text)
        input_ids = self._tokens_to_ids(input_tokens)
        input_ids = self._pad_sequence(input_ids, self.max_length)
        
        # Tokenize output
        output_tokens = ['<START>'] + self._tokenize(output_text) + ['<END>']
        output_ids = self._tokens_to_ids(output_tokens)
        output_ids = self._pad_sequence(output_ids, self.max_length)
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(output_ids, dtype=torch.long)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace."""
        return text.lower().split()
    
    def _tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs using vocabulary."""
        return [self.vocab_to_idx.get(token, self.vocab_to_idx.get('<UNK>', 3)) for token in tokens]
    
    def _pad_sequence(self, sequence: List[int], max_length: int) -> List[int]:
        """Pad sequence to max_length."""
        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            return sequence + [self.pad_token] * (max_length - len(sequence))

def build_vocabulary(training_pairs: List[Tuple[str, str]], min_freq: int = 2) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build vocabulary from training pairs."""
    logger.info("Building vocabulary...")
    
    # Count word frequencies
    word_counts = {}
    for input_text, output_text in training_pairs:
        for text in [input_text, output_text]:
            for word in text.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1
    
    # Create vocabulary with special tokens
    vocab_to_idx = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2,
        '<UNK>': 3
    }
    
    # Add words that appear at least min_freq times
    idx = 4
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab_to_idx[word] = idx
            idx += 1
    
    # Create reverse mapping
    idx_to_vocab = {idx: word for word, idx in vocab_to_idx.items()}
    
    logger.info(f"Built vocabulary with {len(vocab_to_idx)} tokens")
    return vocab_to_idx, idx_to_vocab

def prepare_data_pipeline() -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """Main data preparation pipeline."""
    logger.info("Starting data preparation pipeline...")
    
    # Initialize data collector
    collector = BusinessNameDataCollector()
    
    # Collect and preprocess data
    df = collector.collect_business_names()
    df = collector.preprocess_names(df)
    
    # Create training pairs
    training_pairs = collector.create_training_pairs(df)
    
    # Build vocabulary
    vocab_to_idx, idx_to_vocab = build_vocabulary(training_pairs)
    
    # Split data
    train_pairs, val_pairs = train_test_split(training_pairs, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = BusinessNameDataset(train_pairs, vocab_to_idx)
    val_dataset = BusinessNameDataset(val_pairs, vocab_to_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Save processed data
    processed_data = {
        'vocab_to_idx': vocab_to_idx,
        'idx_to_vocab': idx_to_vocab,
        'train_pairs': train_pairs,
        'val_pairs': val_pairs
    }
    
    torch.save(processed_data, DATA_DIR / "processed" / "business_name_data.pt")
    
    logger.info("Data preparation pipeline completed successfully")
    return train_loader, val_loader, vocab_to_idx, idx_to_vocab
