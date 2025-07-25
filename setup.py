#!/usr/bin/env python3
"""
Setup script for Business Generator ML Pipeline
Automates the installation and configuration process
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description, cwd=None):
    """Run a command and handle errors gracefully."""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            cwd=cwd,
            capture_output=True,
            text=True
        )
        print(f"✅ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return None

def check_prerequisites():
    """Check if required software is installed."""
    print("🔍 Checking prerequisites...")
    
    # Check Python
    try:
        python_version = subprocess.check_output([sys.executable, "--version"], text=True).strip()
        print(f"✅ Python: {python_version}")
    except Exception as e:
        print(f"❌ Python check failed: {e}")
        return False
    
    # Check Node.js
    try:
        node_version = subprocess.check_output(["node", "--version"], text=True).strip()
        print(f"✅ Node.js: {node_version}")
    except Exception as e:
        print(f"❌ Node.js not found. Please install Node.js from https://nodejs.org")
        return False
    
    # Check npm
    try:
        npm_version = subprocess.check_output(["npm", "--version"], text=True).strip()
        print(f"✅ npm: {npm_version}")
    except Exception as e:
        print(f"❌ npm not found: {e}")
        return False
    
    return True

def setup_project():
    """Main setup function."""
    print("🚀 Business Generator ML Pipeline Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please install missing software.")
        return False
    
    project_root = Path(__file__).parent
    
    # Install Node.js dependencies
    if not run_command("npm install", "Installing Node.js dependencies", cwd=project_root):
        return False
    
    # Install concurrently for running both frontend and backend
    if not run_command("npm install concurrently --save-dev", "Installing concurrently", cwd=project_root):
        return False
    
    # Install Python dependencies
    backend_dir = project_root / "backend"
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                      "Installing Python dependencies", cwd=backend_dir):
        return False
    
    # Create necessary directories
    directories_to_create = [
        project_root / "data",
        project_root / "data" / "raw",
        project_root / "data" / "processed",
        project_root / "models",
        project_root / "logs"
    ]
    
    for directory in directories_to_create:
        directory.mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'npm start' to start the application")
    print("2. Run 'python train_pipeline.py' to train the ML models")
    print("3. Visit http://localhost:3000 to use the application")
    
    return True

if __name__ == "__main__":
    success = setup_project()
    sys.exit(0 if success else 1)
