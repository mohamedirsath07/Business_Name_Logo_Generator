import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import json
import pickle
from .config import BUSINESS_NAME_MODEL_CONFIG, MODEL_DIR, LOGGING_CONFIG
import logging.config

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class BusinessNameTransformer(nn.Module):
    """Transformer-based model for business name generation."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, 
                 num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1, max_length: int = 50):
        super(BusinessNameTransformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = self._create_positional_encoding(max_length, embedding_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_length: int, embedding_dim: int) -> torch.Tensor:
        """Create positional encoding for transformer."""
        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                           (-np.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask=None, tgt_mask=None):
        """Forward pass of the transformer."""
        batch_size, src_seq_len = src.shape
        tgt_seq_len = tgt.shape[1]
        
        # Embedding + positional encoding for source
        src_embedded = self.embedding(src) + self.positional_encoding[:, :src_seq_len, :].to(src.device)
        src_embedded = self.dropout(src_embedded)
        
        # Embedding + positional encoding for target
        tgt_embedded = self.embedding(tgt) + self.positional_encoding[:, :tgt_seq_len, :].to(tgt.device)
        tgt_embedded = self.dropout(tgt_embedded)
        
        # Encode source
        memory = self.transformer_encoder(src_embedded, src_key_padding_mask=src_mask)
        
        # Decode target
        output = self.transformer_decoder(tgt_embedded, memory, 
                                        tgt_key_padding_mask=tgt_mask,
                                        memory_key_padding_mask=src_mask)
        
        # Project to vocabulary
        output = self.output_projection(output)
        
        return output
    
    def generate(self, src: torch.Tensor, vocab_to_idx: Dict[str, int], 
                 idx_to_vocab: Dict[int, str], max_length: int = 50) -> str:
        """Generate business name given input."""
        self.eval()
        
        with torch.no_grad():
            batch_size = src.shape[0]
            device = src.device
            
            # Encode source
            src_embedded = self.embedding(src) + self.positional_encoding[:, :src.shape[1], :].to(device)
            memory = self.transformer_encoder(src_embedded)
            
            # Initialize target with START token
            tgt = torch.full((batch_size, 1), vocab_to_idx['<START>'], dtype=torch.long, device=device)
            
            generated_tokens = []
            
            for _ in range(max_length - 1):
                tgt_embedded = self.embedding(tgt) + self.positional_encoding[:, :tgt.shape[1], :].to(device)
                output = self.transformer_decoder(tgt_embedded, memory)
                output = self.output_projection(output)
                
                # Get next token
                next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
                
                # Check for END token
                if next_token.item() == vocab_to_idx['<END>']:
                    break
                
                generated_tokens.append(next_token.item())
                tgt = torch.cat([tgt, next_token], dim=1)
            
            # Convert to text
            text_tokens = [idx_to_vocab.get(token, '<UNK>') for token in generated_tokens]
            return ' '.join(text_tokens)

class BusinessNameTrainer:
    """Trainer class for business name generation model."""
    
    def __init__(self, model: BusinessNameTransformer, vocab_to_idx: Dict[str, int], 
                 idx_to_vocab: Dict[int, str]):
        self.model = model
        self.vocab_to_idx = vocab_to_idx
        self.idx_to_vocab = idx_to_vocab
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab_to_idx['<PAD>'])
        self.optimizer = optim.Adam(model.parameters(), 
                                   lr=BUSINESS_NAME_MODEL_CONFIG['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            # Prepare input and target
            decoder_input = tgt[:, :-1]  # All but last token
            decoder_target = tgt[:, 1:]  # All but first token
            
            # Create masks
            src_padding_mask = (src == self.vocab_to_idx['<PAD>'])
            tgt_padding_mask = (decoder_input == self.vocab_to_idx['<PAD>'])
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(src, decoder_input, src_padding_mask, tgt_padding_mask)
            
            # Calculate loss
            loss = self.criterion(output.reshape(-1, output.size(-1)), 
                                decoder_target.reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                decoder_input = tgt[:, :-1]
                decoder_target = tgt[:, 1:]
                
                src_padding_mask = (src == self.vocab_to_idx['<PAD>'])
                tgt_padding_mask = (decoder_input == self.vocab_to_idx['<PAD>'])
                
                output = self.model(src, decoder_input, src_padding_mask, tgt_padding_mask)
                loss = self.criterion(output.reshape(-1, output.size(-1)), 
                                    decoder_target.reshape(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50):
        """Complete training loop."""
        logger.info(f"Starting training for {epochs} epochs on {self.device}")
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping and model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model(epoch, is_best=True)
                logger.info("New best model saved!")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= BUSINESS_NAME_MODEL_CONFIG['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        logger.info("Training completed!")
        
    def save_model(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'vocab_to_idx': self.vocab_to_idx,
            'idx_to_vocab': self.idx_to_vocab,
            'model_config': BUSINESS_NAME_MODEL_CONFIG
        }
        
        if is_best:
            torch.save(checkpoint, MODEL_DIR / 'best_business_name_model.pt')
        else:
            torch.save(checkpoint, MODEL_DIR / f'business_name_model_epoch_{epoch}.pt')
    
    def load_model(self, checkpoint_path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Model loaded from {checkpoint_path}")

def create_model(vocab_size: int) -> BusinessNameTransformer:
    """Create a new business name generation model."""
    config = BUSINESS_NAME_MODEL_CONFIG
    
    model = BusinessNameTransformer(
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        max_length=config['max_length']
    )
    
    return model

def train_business_name_model(train_loader: DataLoader, val_loader: DataLoader, 
                             vocab_to_idx: Dict[str, int], idx_to_vocab: Dict[int, str]):
    """Main training function for business name model."""
    logger.info("Starting business name model training...")
    
    # Create model
    model = create_model(len(vocab_to_idx))
    
    # Create trainer
    trainer = BusinessNameTrainer(model, vocab_to_idx, idx_to_vocab)
    
    # Train model
    trainer.train(train_loader, val_loader, BUSINESS_NAME_MODEL_CONFIG['epochs'])
    
    logger.info("Business name model training completed!")
    return trainer
