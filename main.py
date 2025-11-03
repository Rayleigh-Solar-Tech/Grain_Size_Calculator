"""
Main entry point for Grain Size Calculator application.
Orchestrates all modules and provides the primary interface.
"""

import sys
import os
import logging
from pathlib import Path

# Add the src directory to Python path so we can import our modules
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import core modules
from core.config import ConfigManager, create_default_config_file
from core.image_processing import load_and_convert_to_grayscale, param_enhance
from core.sam_analysis import create_complete_analyzer
from core.ocr import create_ocr_processor
from core.results import create_complete_results_processor

# Import GUI
from src.frontend.main_window import main as gui_main


def setup_logging():
    """Setup logging configuration."""
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'grain_calculator.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce ultralytics logging
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import ultralytics
    except ImportError:
        missing_deps.append("ultralytics")
    
    try:
        import PyQt5
    except ImportError:
        missing_deps.append("PyQt5")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    # OCR dependencies (at least one should be available)
    ocr_available = False
    try:
        import pytesseract
        ocr_available = True
    except ImportError:
        pass
    
    try:
        import easyocr
        ocr_available = True
    except ImportError:
        pass
    
    if not ocr_available:
        missing_deps.append("pytesseract OR easyocr")
    
    if missing_deps:
        print("Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install missing dependencies using:")
        print("pip install -r requirements.txt")
        return False
    
    return True


def create_default_directories():
    """Create default directories if they don't exist."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    directories = [
        os.path.join(base_dir, 'outputs'),
        os.path.join(base_dir, 'configs'),
        os.path.join(base_dir, 'logs'),
        os.path.join(base_dir, 'temp')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def initialize_config():
    """Initialize configuration system."""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')
    config_file = os.path.join(config_dir, 'default_config.json')
    
    # Create default config if it doesn't exist
    if not os.path.exists(config_file):
        print(f"Creating default configuration at: {config_file}")
        create_default_config_file(config_file)
    
    return config_file


def run_gui():
    """Run the GUI application."""
    print("Starting Grain Size Calculator GUI...")
    gui_main()


def run_cli(image_path, config_file=None, output_dir=None):
    """
    Run command-line interface for batch processing.
    
    Args:
        image_path: Path to SEM image file
        config_file: Path to configuration file (optional)
        output_dir: Output directory (optional)
    """
    print(f"Processing image: {image_path}")
    
    # Load configuration
    config_manager = ConfigManager(config_file)
    if config_file:
        config_manager.load_config()
    
    processing_config = config_manager.processing_config
    variants = config_manager.variants
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(image_path), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load and analyze image
        print("Loading image...")
        gray0, H0, W0 = load_and_convert_to_grayscale(image_path)
        
        # Try to extract scale information using OCR
        print("Detecting scale information...")
        try:
            ocr = create_ocr_processor('auto')
            metadata = ocr.extract_all_metadata(gray0)
            if metadata.get('frame_width_um'):
                frame_width_um = metadata['frame_width_um']
                print(f"Auto-detected frame width: {frame_width_um:.2f} µm")
            else:
                frame_width_um = processing_config.default_frame_width_um
                print(f"Using default frame width: {frame_width_um:.2f} µm")
        except Exception as e:
            print(f"OCR failed, using default frame width: {e}")
            frame_width_um = processing_config.default_frame_width_um
        
        um_per_px = frame_width_um / float(W0)
        print(f"Pixel size: {um_per_px:.4f} µm/px")
        
        # Initialize analysis components
        print("Initializing SAM model...")
        analyzer, feret_calc, measurements = create_complete_analyzer(um_per_pixel=um_per_px)
        
        # Setup results processing
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        processor, exporter, visualizer = create_complete_results_processor(output_dir, base_name)
        
        # Process each variant
        print(f"Processing {len(variants)} variants...")
        variant_results = []
        
        for i, variant in enumerate(variants):
            print(f"  Processing variant {i+1}/{len(variants)}: {variant.name}")
            
            # Enhance image
            gray_enh = param_enhance(
                gray0,
                clip=variant.clip,
                tile=variant.tile,
                gamma=variant.gamma,
                unsharp_amount=variant.unsharp_amount,
                unsharp_sigma=variant.unsharp_sigma
            )
            
            # Prepare for SAM
            from core.image_processing import resize_for_processing, prep_rgb8
            gray_work, scale = resize_for_processing(gray_enh, processing_config.max_side_cpu)
            rgb8 = prep_rgb8(gray_work)
            
            # Run SAM
            sam_results, proc_time = analyzer.segment_grains(rgb8)
            
            # Process results
            area_threshold = processing_config.min_area_px if scale == 1.0 else int(round(processing_config.min_area_px * (scale**2)))
            label_small = analyzer.masks_to_labels(getattr(sam_results[0], "masks", None), min_area=area_threshold)
            
            if label_small is not None and scale != 1.0:
                import cv2
                label_full = cv2.resize(label_small, (W0, H0), interpolation=cv2.INTER_NEAREST)
            else:
                label_full = label_small
            
            # Calculate measurements
            grain_data = []
            if label_full is not None and label_full.max() > 0:
                per_grain = feret_calc.calculate_per_grain_feret(label_full, min_area=area_threshold)
                grain_data = measurements.convert_measurements_to_microns(per_grain)
            
            # Process results
            result = processor.process_variant_results(
                variant.name, grain_data, 
                processing_config.apply_feret_cap, 
                processing_config.feret_cap_um
            )
            variant_results.append(result)
            
            print(f"    Detected {len(grain_data)} grains, {result['grains_used']} used")
            
            # Save overlay if enabled
            if processing_config.save_overlays and grain_data:
                from core.image_processing import normalize01, create_overlay_visualization
                import matplotlib.pyplot as plt
                
                disp01 = normalize01(gray_enh)
                overlay = create_overlay_visualization(
                    disp01, label_full, grain_data, 
                    processing_config.annotate_measurements
                )
                
                overlay_path = os.path.join(output_dir, f"{base_name}_{variant.name}_overlay.png")
                plt.imsave(overlay_path, overlay)
        
        # Combine results and export
        print("Combining results and exporting...")
        combined_results = processor.combine_variant_results(variant_results)
        
        processing_config_dict = {
            'apply_feret_cap': processing_config.apply_feret_cap,
            'feret_cap_um': processing_config.feret_cap_um,
            'min_area_px': processing_config.min_area_px,
            'frame_width_um': frame_width_um,
            'um_per_pixel': um_per_px
        }
        
        exported_files = exporter.export_all_formats(
            variant_results, combined_results, processing_config_dict
        )
        
        print("Results exported:")
        for format_name, file_path in exported_files.items():
            print(f"  {format_name}: {file_path}")
        
        # Create visualizations
        try:
            print("Creating visualizations...")
            plot_files = visualizer.create_distribution_plots(variant_results, base_name)
            comparison_plot = visualizer.create_summary_comparison(variant_results, base_name)
            
            print("Visualizations created:")
            for plot_file in plot_files:
                print(f"  {plot_file}")
            print(f"  {comparison_plot}")
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
        
        print(f"Analysis complete! Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def main():
    """Main application entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Grain Size Calculator - SEM Image Analysis")
    parser.add_argument("--gui", action="store_true", help="Run GUI application (default)")
    parser.add_argument("--cli", metavar="IMAGE_PATH", help="Run CLI mode with specified image")
    parser.add_argument("--config", metavar="CONFIG_FILE", help="Configuration file path")
    parser.add_argument("--output", metavar="OUTPUT_DIR", help="Output directory")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies and exit")
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        if check_dependencies():
            print("All dependencies are available!")
            return 0
        else:
            return 1
    
    # Setup logging and directories
    setup_logging()
    create_default_directories()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Initialize configuration
    config_file = initialize_config()
    
    # Run appropriate interface
    if args.cli:
        # Command-line interface
        return run_cli(args.cli, args.config or config_file, args.output)
    else:
        # GUI interface (default)
        run_gui()
        return 0


if __name__ == "__main__":
    sys.exit(main())