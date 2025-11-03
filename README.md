# Grain Size Calculator

A comprehensive desktop application for analyzing grain sizes in SEM (Scanning Electron Microscope) images using AI-powered segmentation and automated measurements.

## Features

- **AI-Powered Segmentation**: Uses Meta's Segment Anything Model (SAM) for accurate grain detection
- **OCR Scale Detection**: Automatically extracts scale information from SEM image metadata
- **Multiple Analysis Variants**: Process images with different enhancement parameters
- **Comprehensive Measurements**: Calculate Feret diameters, areas, and statistical summaries
- **Modern Desktop GUI**: PyQt5-based interface with real-time progress tracking
- **Batch Processing**: Command-line interface for automated analysis
- **Multiple Export Formats**: CSV, JSON, and visualization outputs

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- For OCR functionality, install Tesseract OCR:
  - **Windows**: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
  - **Linux**: `sudo apt-get install tesseract-ocr`
  - **macOS**: `brew install tesseract`

### Setup Instructions

1. **Clone or Download the Repository**
   ```bash
   git clone https://github.com/Rayleigh-Solar-Tech/Grain_Size_Calculator.git
   cd Grain_Size_Calculator
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   python -m venv grain_calculator_env
   
   # Windows
   grain_calculator_env\Scripts\activate
   
   # Linux/macOS
   source grain_calculator_env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python main.py --check-deps
   ```

### GPU Support (Optional but Recommended)

For faster processing with CUDA support:
```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Desktop GUI Application

Launch the graphical interface:
```bash
python main.py
```

#### GUI Workflow:

1. **Load Image**: Click "Browse Image" to select your SEM image
2. **Scale Detection**: 
   - Enable "Auto-detect scale" to use OCR
   - Or manually enter the frame width in micrometers
3. **Configure Analysis**:
   - Set minimum grain area threshold
   - Enable/disable Feret diameter cap
   - Choose output options
4. **Review Analysis Variants**: The default variants cover different enhancement parameters
5. **Start Analysis**: Click "Start Analysis" and monitor progress
6. **Review Results**: Check the results table and export data

### Command Line Interface

For batch processing or automation:
```bash
python main.py --cli path/to/your/image.tiff --output path/to/output/directory
```

Options:
- `--cli IMAGE_PATH`: Run in command-line mode with specified image
- `--config CONFIG_FILE`: Use custom configuration file
- `--output OUTPUT_DIR`: Specify output directory
- `--check-deps`: Check if all dependencies are installed

## Configuration

The application uses JSON configuration files for customizing analysis parameters:

### Analysis Variants

Each variant defines image enhancement parameters:
```json
{
  "name": "variant_name",
  "clip": 1.3,        // CLAHE clip limit
  "tile": 8,          // CLAHE tile grid size
  "gamma": 1.0,       // Gamma correction
  "unsharp_amount": 0.7,  // Unsharp masking amount
  "unsharp_sigma": 1.0    // Unsharp masking sigma
}
```

### Processing Configuration

Main processing parameters:
```json
{
  "model_gpu": "sam_l.pt",           // SAM model for GPU
  "model_cpu": "sam_l.pt",           // SAM model for CPU
  "min_area_px": 50,                 // Minimum grain area (pixels)
  "apply_feret_cap": true,           // Apply Feret diameter cap
  "feret_cap_um": 5.0,               // Maximum Feret diameter (μm)
  "save_overlays": true,             // Save annotated images
  "default_frame_width_um": 21.8     // Default frame width (μm)
}
```

## Output Files

The application generates several output files:

### CSV Files
- `*_per_variant_chords.csv`: Chord length statistics by variant
- `*_per_variant_areas.csv`: Area statistics by variant  
- `*_overall_averages.csv`: Combined statistics summary

### Visualizations
- `*_chord_distribution.png`: Histogram of chord length distributions
- `*_area_distribution.png`: Histogram of area distributions
- `*_variant_comparison.png`: Comparison chart of variant results
- `*_overlay.png`: Annotated segmentation overlays (per variant)

### Detailed Results
- `*_detailed_results.json`: Complete analysis results with metadata

## Supported Image Formats

- TIFF (.tiff, .tif)
- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)

## System Requirements

### Minimum Requirements
- 8 GB RAM
- 2 GB free disk space
- CPU: Intel Core i5 or AMD Ryzen 5 equivalent

### Recommended Requirements
- 16 GB RAM
- NVIDIA GPU with 6+ GB VRAM
- 5 GB free disk space
- CPU: Intel Core i7 or AMD Ryzen 7 equivalent

## Troubleshooting

### Common Issues

1. **OCR Not Working**
   - Ensure Tesseract is installed and in PATH
   - Try different OCR engines (easyocr vs tesseract)
   - Check if image footer contains readable scale information

2. **SAM Model Loading Errors**
   - Check internet connection (models download automatically)
   - Ensure sufficient disk space
   - Try CPU model if GPU model fails

3. **Memory Issues**
   - Reduce image size or use CPU processing
   - Close other applications
   - Enable image downscaling for CPU processing

4. **Dependencies Missing**
   - Run `python main.py --check-deps`
   - Reinstall requirements: `pip install -r requirements.txt`

### Performance Tips

- Use GPU acceleration when available
- Enable image downscaling for CPU processing
- Process images in batches using CLI mode
- Close unnecessary applications during analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```
Grain Size Calculator - SEM Image Analysis Tool
Rayleigh Solar Tech
https://github.com/Rayleigh-Solar-Tech/Grain_Size_Calculator
```

## Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Contact: [your-email@domain.com]

## Acknowledgments

- Meta AI for the Segment Anything Model (SAM)
- OpenCV community for image processing tools
- PyQt5 team for the GUI framework
- Contributors and testers