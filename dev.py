#!/usr/bin/env python3
"""
Development utilities for Business Generator ML Pipeline
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def start_development():
    """Start the development environment."""
    print("🚀 Starting development environment...")
    
    # Start both frontend and backend
    try:
        subprocess.run(["npm", "start"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start development environment: {e}")
        return False
    
    return True

def train_models():
    """Train the ML models."""
    print("🧠 Training ML models...")
    
    try:
        subprocess.run([sys.executable, "train_pipeline.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False
    
    return True

def run_tests():
    """Run tests for the application."""
    print("🧪 Running tests...")
    
    # Run frontend tests
    try:
        subprocess.run(["npm", "test", "--", "--watchAll=false"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend tests failed: {e}")
        return False
    
    # TODO: Add backend tests
    print("✅ All tests passed!")
    return True

def build_production():
    """Build the application for production."""
    print("🏗️ Building for production...")
    
    try:
        subprocess.run(["npm", "run", "build"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False
    
    print("✅ Production build completed!")
    return True

def clean_project():
    """Clean project files and dependencies."""
    print("🧹 Cleaning project...")
    
    # Remove node_modules
    node_modules = Path("node_modules")
    if node_modules.exists():
        import shutil
        shutil.rmtree(node_modules)
        print("✅ Removed node_modules")
    
    # Remove build directory
    build_dir = Path("build")
    if build_dir.exists():
        import shutil
        shutil.rmtree(build_dir)
        print("✅ Removed build directory")
    
    # Remove Python cache
    for cache_dir in Path(".").rglob("__pycache__"):
        import shutil
        shutil.rmtree(cache_dir)
        print(f"✅ Removed {cache_dir}")
    
    print("🎉 Project cleaned!")
    return True

def show_status():
    """Show project status and health."""
    print("📊 Project Status")
    print("=" * 30)
    
    # Check if dependencies are installed
    node_modules = Path("node_modules")
    print(f"Node modules: {'✅ Installed' if node_modules.exists() else '❌ Not installed'}")
    
    # Check backend requirements
    try:
        import torch
        print("✅ PyTorch installed")
    except ImportError:
        print("❌ PyTorch not installed")
    
    try:
        import pandas
        print("✅ Pandas installed")
    except ImportError:
        print("❌ Pandas not installed")
    
    try:
        import flask
        print("✅ Flask installed")
    except ImportError:
        print("❌ Flask not installed")
    
    # Check if models exist
    models_dir = Path("models")
    if models_dir.exists() and any(models_dir.iterdir()):
        print("✅ Trained models available")
    else:
        print("⚠️ No trained models found - run training first")
    
    return True

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Business Generator Development Tools")
    parser.add_argument("command", choices=[
        "start", "train", "test", "build", "clean", "status"
    ], help="Command to execute")
    
    args = parser.parse_args()
    
    commands = {
        "start": start_development,
        "train": train_models,
        "test": run_tests,
        "build": build_production,
        "clean": clean_project,
        "status": show_status
    }
    
    success = commands[args.command]()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
