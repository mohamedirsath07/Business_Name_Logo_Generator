#!/usr/bin/env python3
"""
Quick training script for the Business Generator ML Pipeline.
Uses lightweight models for fast deployment.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    """Main training function."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("="*50)
    logger.info("🚀 Business Generator ML Pipeline Training")
    logger.info("="*50)
    
    try:
        # Import and initialize pipeline
        from ml_pipeline.simple_models import SimplePipeline
        pipeline = SimplePipeline()
        
        # Train the pipeline (this is fast!)
        logger.info("🧠 Starting lightweight ML pipeline training...")
        result = pipeline.train_pipeline()
        
        if result["status"] == "success":
            logger.info("✅ Training completed successfully!")
            
            # Quick evaluation
            logger.info("📊 Evaluating pipeline...")
            eval_results = pipeline.evaluate_pipeline()
            
            logger.info(f"📈 Success rate: {eval_results.get('success_rate', 0):.1%}")
            logger.info(f"✔️  Tests passed: {eval_results.get('passed_tests', 0)}/{eval_results.get('total_tests', 0)}")
            
            # Test generation
            logger.info("🧪 Testing name generation...")
            test_names = pipeline.generate_business_names("coffee shop", "modern")
            logger.info(f"Generated names: {test_names[:3]}...")
            
            logger.info("🎉 Pipeline is ready for use!")
            
        else:
            logger.error(f"❌ Training failed: {result.get('message', 'Unknown error')}")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"💥 Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
