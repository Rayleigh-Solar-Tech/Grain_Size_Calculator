"""
Analysis worker thread for grain size calculations.
Handles the heavy processing in a separate thread to keep UI responsive.
"""

import os
import time
import traceback
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

# Import our core modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.image_processing import (load_and_convert_to_grayscale, param_enhance, 
                                 prep_rgb8, resize_for_processing, create_overlay_visualization)
from core.sam_analysis import create_complete_analyzer
from core.results import create_complete_results_processor
from core.pdf_report import create_pdf_report_generator
from core.config import ProcessingConfig
import matplotlib.pyplot as plt


class AnalysisWorker(QThread):
    """Worker thread for performing grain analysis."""
    
    # Signals
    progress_updated = pyqtSignal(int, str)  # progress_value, status_message
    log_message = pyqtSignal(str)  # log_message
    analysis_completed = pyqtSignal(dict)  # results_dict
    error_occurred = pyqtSignal(str)  # error_message
    
    def __init__(self, analysis_params):
        """
        Initialize analysis worker.
        
        Args:
            analysis_params: Dictionary with analysis parameters
        """
        super().__init__()
        self.params = analysis_params
        self.should_stop = False
        self.results = {}
    
    def stop(self):
        """Stop the analysis."""
        self.should_stop = True
    
    def run(self):
        """Main analysis execution."""
        try:
            self.perform_analysis()
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
    
    def perform_analysis(self):
        """Perform the complete grain analysis."""
        # Extract parameters
        image_path = self.params['image_path']
        frame_width_um = self.params['frame_width_um']
        min_area_px = self.params['min_area_px']
        apply_feret_cap = self.params['apply_feret_cap']
        feret_cap_um = self.params['feret_cap_um']
        save_all_data = self.params.get('save_all_data', False)
        save_overlays = self.params['save_overlays']
        annotate_measurements = self.params['annotate_measurements']
        variants = self.params['variants']
        
        # Ridge filtering parameters (with defaults for backward compatibility)
        apply_ridge_filtering = self.params.get('apply_ridge_filtering', False)
        ridge_threshold = self.params.get('ridge_threshold', 0.15)
        ridge_tv_weight = self.params.get('ridge_tv_weight', 0.01)
        ridge_percentile = self.params.get('ridge_percentile', 70)
        ridge_min_size = self.params.get('ridge_min_size', 50)
        
        self.log_message.emit(f"Starting analysis of: {os.path.basename(image_path)}")
        self.progress_updated.emit(5, "Loading image...")
        
        if self.should_stop:
            return
        
        # Load and prepare image
        gray0, H0, W0 = load_and_convert_to_grayscale(image_path)
        um_per_px = frame_width_um / float(W0)
        
        self.log_message.emit(f"Image dimensions: {W0}x{H0}")
        self.log_message.emit(f"Pixel size: {um_per_px:.4f} µm/px")
        
        # Setup output directories - create separate folder for each file
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.join(os.path.dirname(image_path), f"{base_name}_output")
        os.makedirs(out_dir, exist_ok=True)
        
        self.log_message.emit(f"📁 Output folder: {base_name}_output")
        
        self.progress_updated.emit(10, "Initializing SAM model...")
        
        # Get tiling config from params (with defaults)
        enable_tiling = self.params.get('enable_tiling', True)
        tile_size = self.params.get('tile_size', 1024)
        tile_overlap = self.params.get('tile_overlap', 128)
        min_image_size_for_tiling = self.params.get('min_image_size_for_tiling', 2048)
        model_gpu = self.params.get('model_gpu', 'sam_b.pt')
        model_cpu = self.params.get('model_cpu', 'sam_b.pt')
        
        # Initialize analysis components with explicit model names
        analyzer, feret_calc, measurements = create_complete_analyzer(
            model_gpu=model_gpu,
            model_cpu=model_cpu, 
            device=None,  # Auto-detect
            um_per_pixel=um_per_px,
            enable_tiling=enable_tiling,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            min_image_size_for_tiling=min_image_size_for_tiling
        )
        
        # Log device information
        try:
            device_info = analyzer.get_device_info()
            device_type = device_info.get("device_type", "Unknown")
            if device_type == "GPU":
                gpu_name = device_info.get("gpu_name", "Unknown GPU")
                gpu_memory = device_info.get("gpu_memory", 0)
                self.log_message.emit(f"Using GPU acceleration: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                cpu_cores = device_info.get("cpu_cores", 0)
                cpu_threads = device_info.get("cpu_threads", 0)
                self.log_message.emit(f"Using CPU processing: {cpu_cores} cores, {cpu_threads} threads")
        except Exception as e:
            self.log_message.emit(f"Device info warning: {str(e)}")
        
        # Setup results processing
        processor, exporter, visualizer = create_complete_results_processor(
            out_dir, base_name
        )
        
        self.progress_updated.emit(15, "Starting variant analysis...")
        
        # Process each variant
        variant_results = []
        overlay_paths = {}  # Track overlay paths for PDF report
        total_variants = len(variants)
        
        for i, variant in enumerate(variants):
            if self.should_stop:
                return
            
            variant_name = variant.name
            progress = 15 + (i * 70) // total_variants
            self.progress_updated.emit(progress, f"Processing variant: {variant_name}")
            self.log_message.emit(f"Processing variant {i+1}/{total_variants}: {variant_name}")
            
            try:
                # Enhance image with variant parameters
                gray_enh = param_enhance(
                    gray0, 
                    clip=variant.clip,
                    tile=variant.tile,
                    gamma=variant.gamma,
                    unsharp_amount=variant.unsharp_amount,
                    unsharp_sigma=variant.unsharp_sigma
                )
                
                # Prepare for SAM processing - high resolution for best accuracy
                gray_work, scale = resize_for_processing(gray_enh, 1536)  # Higher resolution for better results
                rgb8 = prep_rgb8(gray_work)
                
                # Run SAM segmentation with optional ridge filtering
                self.log_message.emit(f"  Running SAM segmentation...")
                start_time = time.time()
                
                # Prepare ridge config if enabled
                ridge_config = None
                if apply_ridge_filtering:
                    ridge_config = {
                        'ridge_threshold': ridge_threshold,
                        'tv_weight': ridge_tv_weight,
                        'ridge_percentile': ridge_percentile,
                        'min_size': ridge_min_size
                    }
                    self.log_message.emit(f"  Ridge filtering enabled (threshold={ridge_threshold})")
                
                sam_results, processing_time, filtered_data = analyzer.segment_grains(
                    rgb8, 
                    apply_ridge_filtering=apply_ridge_filtering,
                    ridge_config=ridge_config
                )
                self.log_message.emit(f"  SAM processing completed in {processing_time:.2f}s")
                
                # Convert masks to labels - use ONLY filtered masks if ridge filtering is enabled
                area_threshold = min_area_px if scale == 1.0 else int(round(min_area_px * (scale**2)))
                
                if apply_ridge_filtering and filtered_data is not None:
                    # Use ONLY the accepted/colored grains
                    total_sam_masks = sam_results[0].masks.data.shape[0] if hasattr(sam_results[0].masks, 'data') else 0
                    self.log_message.emit(f"  Ridge filtering: {len(filtered_data['accepted_masks'])} grains kept (of {total_sam_masks} SAM masks)")
                    self.log_message.emit(f"  Unique grains by color: {filtered_data['unique_grain_count']}")
                    label_small = analyzer.create_label_from_masks(filtered_data['accepted_masks'], min_area=area_threshold)
                else:
                    # Use all SAM masks (backward compatibility)
                    label_small = analyzer.masks_to_labels(
                        getattr(sam_results[0], "masks", None), 
                        min_area=area_threshold
                    )
                
                # Resize labels back to original size if needed
                if label_small is not None and scale != 1.0:
                    import cv2
                    label_full = cv2.resize(label_small, (W0, H0), interpolation=cv2.INTER_NEAREST)
                else:
                    label_full = label_small
                
                grains_detected = int(label_full.max()) if (label_full is not None and label_full.size) else 0
                
                # Calculate grain measurements
                grain_data = []
                if grains_detected > 0:
                    per_grain = feret_calc.calculate_per_grain_feret(label_full, min_area=area_threshold)
                    grain_data = measurements.convert_measurements_to_microns(per_grain)
                
                # Process results for this variant
                result = processor.process_variant_results(
                    variant_name, grain_data, apply_feret_cap, feret_cap_um
                )
                variant_results.append(result)
                
                self.log_message.emit(f"  ✓ Final grain count: {result['grains_used']} grains (after size/quality filtering)")
                
                # Save ridge-filtered colored image if ridge filtering was used and grains were detected (only if save_all_data)
                if save_all_data and filtered_data is not None and 'colored_image' in filtered_data and len(filtered_data.get('accepted_masks', [])) > 0:
                    try:
                        # Ensure output directory exists
                        os.makedirs(out_dir, exist_ok=True)
                        # Normalize path to avoid mixed separators
                        ridge_output_path = os.path.normpath(os.path.join(out_dir, f"{base_name}_{variant_name}_ridge_filtered.png"))
                        # Convert colored_image to proper format for saving
                        colored_img = filtered_data['colored_image']
                        if colored_img.dtype == np.uint8:
                            colored_img = colored_img.astype(np.float32) / 255.0
                        plt.imsave(ridge_output_path, colored_img)
                        self.log_message.emit(f"  Ridge-filtered image saved: {os.path.basename(ridge_output_path)}")
                    except Exception as e:
                        self.log_message.emit(f"  Warning: Could not save ridge-filtered image: {e}")
                
                # Save overlay if requested (always save for PDF, store path)
                if grains_detected > 0:
                    try:
                        from core.image_processing import normalize01
                        disp01 = normalize01(gray_enh)
                        overlay = create_overlay_visualization(
                            disp01, label_full, grain_data, annotate_measurements
                        )
                        
                        # Normalize path to avoid mixed separators
                        overlay_path = os.path.normpath(os.path.join(out_dir, f"{base_name}_{variant_name}_overlay.png"))
                        os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
                        plt.imsave(overlay_path, overlay)
                        overlay_paths[variant_name] = overlay_path  # Track for PDF report
                        self.log_message.emit(f"  Overlay saved: {os.path.basename(overlay_path)}")
                    except Exception as e:
                        self.log_message.emit(f"  Warning: Could not save overlay: {e}")
                
            except Exception as e:
                error_msg = f"Error processing variant {variant_name}: {str(e)}"
                self.log_message.emit(error_msg)
                # Continue with next variant
                continue
        
        if self.should_stop:
            return
        
        self.progress_updated.emit(85, "Combining results...")
        
        # Combine all variant results
        combined_results = processor.combine_variant_results(variant_results)
        
        self.log_message.emit(f"Analysis completed: {combined_results['total_grains_pooled']} total grains processed")
        
        # Export results
        self.progress_updated.emit(90, "Exporting results...")
        
        processing_config_dict = {
            'apply_feret_cap': apply_feret_cap,
            'feret_cap_um': feret_cap_um,
            'min_area_px': min_area_px,
            'frame_width_um': frame_width_um,
            'um_per_pixel': um_per_px
        }
        
        # Always export CSV files (needed for PDF), but will be deleted if save_all_data=False
        exported_files = exporter.export_all_formats(
            variant_results, 
            combined_results, 
            processing_config_dict
        )
        
        if save_all_data:
            self.log_message.emit("Results exported:")
            for format_name, file_path in exported_files.items():
                self.log_message.emit(f"  {format_name}: {os.path.basename(file_path)}")
        
        # Create visualizations only if save_all_data is enabled
        if save_all_data:
            self.progress_updated.emit(95, "Creating visualizations...")
            
            try:
                plot_files = visualizer.create_distribution_plots(variant_results, base_name)
                comparison_plot = visualizer.create_summary_comparison(variant_results, base_name)
                
                self.log_message.emit("Visualizations created:")
                for plot_file in plot_files:
                    self.log_message.emit(f"  {os.path.basename(plot_file)}")
                self.log_message.emit(f"  {os.path.basename(comparison_plot)}")
                
            except Exception as e:
                self.log_message.emit(f"Warning: Could not create visualizations: {str(e)}")
        
        # Create comprehensive PDF report
        self.progress_updated.emit(97, "Generating PDF report...")
        
        try:
            pdf_generator = create_pdf_report_generator(out_dir, base_name)
            
            metadata_dict = {
                'image_path': image_path,
                'image_dimensions': (W0, H0),
                'um_per_pixel': um_per_px,
                'frame_width_um': frame_width_um,
                'total_variants': len(variants)
            }
            
            self.log_message.emit(f"Creating PDF with {len(exported_files)} CSV files...")
            print(f"DEBUG: exported_files keys = {list(exported_files.keys())}")
            
            pdf_path = pdf_generator.create_complete_report(
                variant_results=variant_results,
                combined_results=combined_results,
                processing_config=processing_config_dict,
                metadata=metadata_dict,
                overlay_paths=overlay_paths,
                exported_files=exported_files
            )
            
            self.log_message.emit(f"📄 PDF Report created: {os.path.basename(pdf_path)}")
            
            # Clean up temporary files if not saving all data
            if not save_all_data:
                # Delete overlay images
                for overlay_path in overlay_paths.values():
                    try:
                        if os.path.exists(overlay_path):
                            os.remove(overlay_path)
                    except Exception as e:
                        pass  # Silently ignore cleanup errors
                
                # Delete CSV files
                for file_path in exported_files.values():
                    try:
                        if os.path.exists(file_path) and file_path.endswith('.csv'):
                            os.remove(file_path)
                    except Exception as e:
                        pass  # Silently ignore cleanup errors
                
                self.log_message.emit("Temporary files cleaned up (only PDF saved)")
            
        except Exception as e:
            self.log_message.emit(f"Warning: Could not create PDF report: {str(e)}")
        
        # Prepare final results
        final_results = {
            'variant_results': variant_results,
            'combined_results': combined_results,
            'processing_config': processing_config_dict,
            'exported_files': exported_files,
            'metadata': {
                'image_path': image_path,
                'image_dimensions': (W0, H0),
                'um_per_pixel': um_per_px,
                'total_variants': len(variants),
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        self.progress_updated.emit(100, "Analysis complete!")
        self.analysis_completed.emit(final_results)