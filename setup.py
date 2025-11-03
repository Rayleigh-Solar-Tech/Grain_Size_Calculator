"""
Setup script for Grain Size Calculator.
Helps with installation and initial configuration.
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    return True


def install_requirements():
    """Install required packages."""
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"Error: {requirements_file} not found.")
        return False
    
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        "outputs",
        "configs", 
        "logs",
        "temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def create_default_config():
    """Create default configuration file."""
    config_dir = "configs"
    config_file = os.path.join(config_dir, "default_config.json")
    
    default_config = {
        "processing": {
            "model_gpu": "sam_l.pt",
            "model_cpu": "sam_l.pt", 
            "max_side_cpu": 1536,
            "min_area_px": 50,
            "apply_feret_cap": True,
            "feret_cap_um": 5.0,
            "save_overlays": True,
            "save_contact_sheet": False,
            "annotate_measurements": True,
            "default_frame_width_um": 21.8
        },
        "variants": [
            {
                "name": "v22_cl1.3_t12_g0.98_us0.70_s1.1",
                "clip": 1.3,
                "tile": 12,
                "gamma": 0.98,
                "unsharp_amount": 0.70,
                "unsharp_sigma": 1.1
            },
            {
                "name": "v14_cl1.3_t10_g1.00_us0.70_s1.2",
                "clip": 1.3,
                "tile": 10,
                "gamma": 1.00,
                "unsharp_amount": 0.70,
                "unsharp_sigma": 1.2
            },
            {
                "name": "v5_cl1.5_t8_g1.00_us0.80_s1.0",
                "clip": 1.5,
                "tile": 8,
                "gamma": 1.00,
                "unsharp_amount": 0.80,
                "unsharp_sigma": 1.0
            },
            {
                "name": "v10_cl1.5_t8_g0.98_us0.80_s1.0",
                "clip": 1.5,
                "tile": 8,
                "gamma": 0.98,
                "unsharp_amount": 0.80,
                "unsharp_sigma": 1.0
            },
            {
                "name": "v4_cl3_t8_g0.85_us1.50_s1.2",
                "clip": 3.0,
                "tile": 8,
                "gamma": 0.85,
                "unsharp_amount": 1.50,
                "unsharp_sigma": 1.2
            },
            {
                "name": "v2_cl2.5_t8_g0.95_us1.20_s1.0",
                "clip": 2.5,
                "tile": 8,
                "gamma": 0.95,
                "unsharp_amount": 1.20,
                "unsharp_sigma": 1.0
            }
        ]
    }
    
    with open(config_file, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"Created default configuration: {config_file}")


def check_tesseract():
    """Check if Tesseract is installed."""
    try:
        subprocess.run(["tesseract", "--version"], 
                      capture_output=True, check=True)
        print("✓ Tesseract OCR is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ Tesseract OCR not found. OCR functionality will use EasyOCR only.")
        print("  To install Tesseract:")
        print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Linux: sudo apt-get install tesseract-ocr")
        print("  macOS: brew install tesseract")
        return False


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA GPU available: {gpu_name} ({gpu_count} devices)")
            return True
        else:
            print("⚠ No CUDA GPU detected. Will use CPU processing.")
            return False
    except ImportError:
        print("⚠ PyTorch not installed. Cannot check GPU status.")
        return False


def run_dependency_check():
    """Run the application's dependency check."""
    try:
        result = subprocess.run([sys.executable, "main.py", "--check-deps"], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ All application dependencies are satisfied")
            return True
        else:
            print("✗ Some dependencies are missing:")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"Error checking dependencies: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("Grain Size Calculator - Setup Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Install requirements
    print("\nInstalling Python requirements...")
    if not install_requirements():
        print("Setup failed during requirements installation.")
        return 1
    
    # Create default configuration
    print("\nCreating default configuration...")
    create_default_config()
    
    # Check optional dependencies
    print("\nChecking optional dependencies...")
    check_tesseract()
    check_gpu()
    
    # Run application dependency check
    print("\nRunning application dependency check...")
    if not run_dependency_check():
        print("Warning: Some dependencies may be missing.")
    
    print("\n" + "=" * 60)
    print("Setup completed!")
    print("=" * 60)
    print("\nTo run the application:")
    print("  GUI mode: python main.py")
    print("  CLI mode: python main.py --cli path/to/image.tiff")
    print("  Help:     python main.py --help")
    print("\nFor detailed usage instructions, see README.md")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())