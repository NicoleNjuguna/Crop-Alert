#!/usr/bin/env python3
"""
Setup and Installation Script for Maize Disease Alert System
Validates the installation and provides setup guidance.

Usage: python setup.py
"""

import sys
import subprocess
import importlib
import platform

def check_python_version():
    """Check if Python version is 3.9 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9 or higher is required")
        print(f"Current version: {platform.python_version()}")
        return False
    else:
        print(f"✅ Python version: {platform.python_version()}")
        return True

def install_requirements():
    """Install required packages from requirements.txt."""
    try:
        print("📦 Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def check_imports():
    """Check if all required packages can be imported."""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'folium',
        'PIL',
        'cv2',
        'tensorflow'
    ]
    
    print("🔍 Checking package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'cv2':
                importlib.import_module('cv2')
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {str(e)}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def validate_file_structure():
    """Check if all required files are present."""
    import os
    
    required_files = [
        'app.py',
        'utils.py', 
        'requirements.txt',
        'README.md'
    ]
    
    print("📁 Checking file structure...")
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)
    
    return len(missing_files) == 0

def run_basic_syntax_check():
    """Run basic syntax check on Python files."""
    import ast
    
    files_to_check = ['app.py', 'utils.py']
    
    print("🔍 Running syntax validation...")
    
    for file in files_to_check:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the file to check for syntax errors
            ast.parse(content)
            print(f"✅ {file} - Syntax OK")
            
        except SyntaxError as e:
            print(f"❌ {file} - Syntax Error: {str(e)}")
            return False
        except FileNotFoundError:
            print(f"❌ {file} - File not found")
            return False
        except Exception as e:
            print(f"⚠️ {file} - Warning: {str(e)}")
    
    return True

def main():
    """Main setup and validation function."""
    print("🌽 Maize Disease Alert System - Setup & Validation")
    print("=" * 50)
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Validate file structure
    if not validate_file_structure():
        print("\n❌ Missing required files. Please ensure all files are present.")
        sys.exit(1)
    
    # Step 3: Run syntax check
    if not run_basic_syntax_check():
        print("\n❌ Syntax errors detected. Please fix before proceeding.")
        sys.exit(1)
    
    # Step 4: Ask user if they want to install requirements
    print("\n" + "=" * 50)
    install_deps = input("📦 Install dependencies from requirements.txt? (y/N): ").lower().strip()
    
    if install_deps in ['y', 'yes']:
        if not install_requirements():
            sys.exit(1)
        
        # Step 5: Check imports after installation
        if not check_imports():
            print("\n❌ Some packages failed to import. Please check the installation.")
            sys.exit(1)
    else:
        print("⚠️ Skipping dependency installation")
    
    # Final success message
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
    print("2. Run the application: streamlit run app.py")
    print("3. Open your browser to http://localhost:8501")
    print("\n🔗 For deployment to Streamlit Cloud:")
    print("1. Push this repository to GitHub")
    print("2. Connect your GitHub account to Streamlit Cloud")
    print("3. Deploy with app.py as the main file")
    print("\n📖 Read README.md for detailed instructions")

if __name__ == "__main__":
    main()