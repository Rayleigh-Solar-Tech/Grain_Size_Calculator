"""
SAM (Segment Anything Model) integration for grain analysis.
Handles model loading, segmentation, and grain measurement calculations.
"""

import logging
import math
import time
import cv2
import numpy as np
import torch
from ultralytics import SAM


class GrainAnalyzer:
    """Main class for grain analysis using SAM model."""
    
    def __init__(self, model_gpu="sam_l.pt", model_cpu="sam_l.pt", device=None):
        """
        Initialize the grain analyzer.
        
        Args:
            model_gpu: SAM model for GPU processing
            model_cpu: SAM model for CPU processing
            device: Device to use ('cuda:0', 'cpu', or None for auto-detect)
        """
        self.model_gpu = model_gpu
        self.model_cpu = model_cpu
        
        # Auto-detect device if not specified
        if device is None:
            self.device = self._detect_best_device()
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Optimize for the selected device
        self._configure_device_optimizations()
        
        # Optimize for the selected device
        self._configure_device_optimizations()
        
        # Load model - keep it simple and fast like original code
        model_name = self.model_gpu if self.device.startswith("cuda") else self.model_cpu
        logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
        
        print(f"Loading SAM model: {model_name}")
        self.model = SAM(model_name)
    
    def _detect_best_device(self):
        """
        Force GPU usage like the original fast code - no complex testing.
        """
        # Use the exact same logic as the original working code
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Using device: {device}")
        return device
    
    def _configure_device_optimizations(self):
        """Configure optimizations based on the selected device."""
        if self.device == "cpu":
            self._configure_cpu_optimizations()
        else:
            self._configure_gpu_optimizations()
    
    def _configure_cpu_optimizations(self):
        """Configure optimizations for CPU processing."""
        try:
            import os
            # Use all available CPU cores for best performance
            num_threads = os.cpu_count() or 8
            torch.set_num_threads(num_threads)
            print(f"Using {num_threads} CPU threads for maximum performance")
            
            # Set additional optimizations for CPU inference
            torch.set_grad_enabled(False)  # Disable gradients for inference
            
            # Enable MKL-DNN if available (Intel Math Kernel Library) - Windows compatible
            try:
                torch.backends.mkldnn.enabled = True
                print("MKL-DNN optimization enabled")
            except Exception:
                pass
                
        except Exception as e:
            print(f"CPU optimization warning: {e}")
    
    def _configure_gpu_optimizations(self):
        """Minimal GPU optimizations - keep it simple."""
        print(f"GPU ready for processing")
    
    def get_device_info(self):
        """
        Get information about the current device being used.
        
        Returns:
            dict: Device information
        """
        info = {
            "device": self.device,
            "device_type": "GPU" if self.device.startswith("cuda") else "CPU"
        }
        
        if self.device.startswith("cuda"):
            try:
                gpu_id = int(self.device.split(":")[1])
                info["gpu_name"] = torch.cuda.get_device_name(gpu_id)
                info["gpu_memory"] = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                info["cuda_version"] = torch.version.cuda
            except Exception:
                pass
        else:
            import os
            info["cpu_threads"] = torch.get_num_threads()
            info["cpu_cores"] = os.cpu_count()
            
        return info
    
    def segment_grains(self, rgb_image, verbose=False):
        """
        Segment grains using SAM model.
        
        Args:
            rgb_image: RGB image (uint8, HxWx3)
            verbose: Whether to print verbose output
            
        Returns:
            tuple: (segmentation_results, processing_time)
        """
        start_time = time.time()
        
        # Predict - use exact same parameters as original fast code
        results = self.model.predict(
            rgb_image, 
            device=self.device, 
            verbose=verbose,
            half=self.device.startswith("cuda")
        )
        
        processing_time = time.time() - start_time
        
        # Clean up GPU memory after processing
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            
        return results, processing_time
    
    def masks_to_labels(self, mask_obj, min_area=0):
        """
        Convert SAM masks to label image.
        
        Args:
            mask_obj: SAM mask object
            min_area: Minimum area threshold in pixels
            
        Returns:
            Label image (numpy array) or None if no valid masks
        """
        if mask_obj is None or getattr(mask_obj, "data", None) is None:
            return None
            
        try:
            m_np = mask_obj.data.detach().cpu().numpy()
        except Exception:
            m_np = mask_obj.data.cpu().numpy()
            
        if m_np.ndim != 3:
            return None
            
        n, h, w = m_np.shape
        label_img = np.zeros((h, w), dtype=np.int32)
        
        idx = 1
        for i in range(n):
            mask = (m_np[i] > 0.5)
            if min_area > 0 and mask.sum() < min_area:
                continue
            label_img[mask] = idx
            idx += 1
            
        return label_img


class FeretCalculator:
    """Calculate Feret diameter (maximum caliper diameter) for grain measurements."""
    
    @staticmethod
    def feret_from_contour_points(points):
        """
        Calculate Feret diameter from contour points.
        
        Args:
            points: Contour points as numpy array
            
        Returns:
            tuple: (feret_diameter, point1, point2)
        """
        if points.shape[0] < 2:
            return 0.0, (0, 0), (0, 0)
        
        # Get convex hull
        hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
        if hull.shape[0] < 2:
            return 0.0, (0, 0), (0, 0)
        
        # Limit hull points for performance
        max_hull_pts = 2000
        if hull.shape[0] > max_hull_pts:
            step = int(math.ceil(hull.shape[0] / max_hull_pts))
            hull = hull[::step]
        
        # Calculate all pairwise distances
        A = hull[:, None, :]  # Shape: (N, 1, 2)
        B = hull[None, :, :]  # Shape: (1, N, 2)
        D2 = ((A - B) ** 2).sum(axis=2)  # Shape: (N, N)
        
        # Find maximum distance
        idx = np.unravel_index(np.argmax(D2), D2.shape)
        p1 = tuple(map(float, hull[idx[0]]))
        p2 = tuple(map(float, hull[idx[1]]))
        
        return float(np.sqrt(D2[idx])), p1, p2
    
    def calculate_per_grain_feret(self, label_image, min_area=0):
        """
        Calculate Feret diameter for each grain in label image.
        
        Args:
            label_image: Label image with grain IDs
            min_area: Minimum area threshold
            
        Returns:
            List of grain measurement dictionaries
        """
        results = []
        
        if label_image is None or label_image.max() == 0:
            return results
        
        max_id = int(label_image.max())
        
        for grain_id in range(1, max_id + 1):
            mask = (label_image == grain_id).astype(np.uint8)
            area = int(mask.sum())
            
            if min_area and area < min_area:
                continue
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            
            # Use largest contour
            contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
            
            # Calculate Feret diameter
            length_px, p1, p2 = self.feret_from_contour_points(contour)
            
            results.append({
                "id": grain_id,
                "length_px": length_px,
                "p1": p1,
                "p2": p2,
                "area_px": area
            })
        
        return results


class GrainMeasurements:
    """Handle grain measurements and conversions."""
    
    def __init__(self, um_per_pixel):
        """
        Initialize with pixel to micron conversion factor.
        
        Args:
            um_per_pixel: Microns per pixel conversion factor
        """
        self.um_per_pixel = um_per_pixel
    
    def convert_measurements_to_microns(self, grain_data):
        """
        Convert pixel measurements to microns and add to grain data.
        
        Args:
            grain_data: List of grain measurement dictionaries
            
        Returns:
            Updated grain data with micron measurements
        """
        for grain in grain_data:
            grain["length_um"] = grain["length_px"] * self.um_per_pixel
            grain["area_um2"] = grain["area_px"] * (self.um_per_pixel ** 2)
        
        return grain_data
    
    def apply_size_filter(self, grain_data, max_feret_um=None):
        """
        Filter grains based on size criteria.
        
        Args:
            grain_data: List of grain measurement dictionaries
            max_feret_um: Maximum Feret diameter in microns
            
        Returns:
            Filtered grain data
        """
        if max_feret_um is None:
            return grain_data
        
        return [grain for grain in grain_data if grain.get("length_um", 0) <= max_feret_um]
    
    def extract_measurements(self, grain_data):
        """
        Extract measurement arrays from grain data.
        
        Args:
            grain_data: List of grain measurement dictionaries
            
        Returns:
            tuple: (chord_lengths_um, areas_um2)
        """
        chords_um = [grain.get("length_um", 0) for grain in grain_data]
        areas_um2 = [grain.get("area_um2", 0) for grain in grain_data]
        
        return chords_um, areas_um2


def create_complete_analyzer(model_gpu="sam_l.pt", model_cpu="sam_l.pt", 
                           device=None, um_per_pixel=1.0):
    """
    Create a complete grain analysis setup.
    
    Args:
        model_gpu: SAM model for GPU
        model_cpu: SAM model for CPU
        device: Device to use
        um_per_pixel: Pixel to micron conversion
        
    Returns:
        tuple: (GrainAnalyzer, FeretCalculator, GrainMeasurements)
    """
    analyzer = GrainAnalyzer(model_gpu, model_cpu, device)
    feret_calc = FeretCalculator()
    measurements = GrainMeasurements(um_per_pixel)
    
    return analyzer, feret_calc, measurements