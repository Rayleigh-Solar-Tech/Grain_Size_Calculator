"""
SAM (Segment Anything Model) integration for grain analysis.
Handles model loading, segmentation, and grain measurement calculations.

Enhanced with:
- Ridge-based filtering for accurate grain detection
- Auto-detection of bright/dark grain boundaries  
- Unique grain counting by color
- Anti-overwrite visualization
"""

import logging
import math
import time
import cv2
import numpy as np
import torch
from ultralytics import SAM

# Ridge detection dependencies
try:
    from skimage.restoration import denoise_tv_chambolle
    from skimage.filters import sato
    from skimage.morphology import skeletonize, remove_small_objects
    RIDGE_FILTERING_AVAILABLE = True
except ImportError:
    RIDGE_FILTERING_AVAILABLE = False
    print("Warning: scikit-image not available. Ridge filtering will be disabled.")


class GrainAnalyzer:
    """Main class for grain analysis using SAM model."""
    
    def __init__(self, model_gpu="sam_b.pt", model_cpu="sam_b.pt", device=None, 
                 enable_tiling=True, tile_size=1024, tile_overlap=128, min_image_size_for_tiling=2048):
        """
        Initialize the grain analyzer.
        
        Args:
            model_gpu: SAM model for GPU processing (default: sam_b.pt for precision)
            model_cpu: SAM model for CPU processing (default: sam_b.pt for precision)
            device: Device to use ('cuda:0', 'cpu', or None for auto-detect)
            enable_tiling: Enable automatic tiling for large images
            tile_size: Size of each tile in pixels
            tile_overlap: Overlap between tiles to avoid edge artifacts
            min_image_size_for_tiling: Only tile images larger than this
        """
        self.model_gpu = model_gpu
        self.model_cpu = model_cpu
        
        # Tiling configuration
        self.enable_tiling = enable_tiling
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.min_image_size_for_tiling = min_image_size_for_tiling
        
        # Auto-detect device if not specified
        if device is None:
            self.device = self._detect_best_device()
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Optimize for the selected device
        self._configure_device_optimizations()
        
        # Load model - keep it simple and fast like original code
        model_name = self.model_gpu if self.device.startswith("cuda") else self.model_cpu
        logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
        
        print(f"Loading SAM model: {model_name}")
        if enable_tiling:
            print(f"Tiling enabled: {tile_size}x{tile_size} px tiles for images >{min_image_size_for_tiling} px")
        self.model = SAM(model_name)
    
    def _detect_best_device(self):
        """
        Force GPU usage like the original fast code - no complex testing.
        PyInstaller-safe device detection.
        """
        try:
            # Check if running as PyInstaller executable
            import sys
            is_executable = getattr(sys, 'frozen', False)
            
            if is_executable:
                print("🔧 Running as executable - Attempting GPU detection...")
            
            # Always try GPU detection first, regardless of executable status
            if torch.cuda.is_available():
                try:
                    print(f"🔍 CUDA available, testing GPU functionality...")
                    # Test GPU with a simple operation
                    test_tensor = torch.tensor([1.0]).cuda()
                    _ = test_tensor + 1
                    device = "cuda:0"
                    
                    if is_executable:
                        print(f"� SUCCESS: GPU working in executable! Using: {device}")
                    else:
                        print(f"🚀 GPU verified working in script mode: {device}")
                    
                    return device
                    
                except Exception as e:
                    if is_executable:
                        print(f"⚠️ GPU test failed in executable: {e}")
                        print("💡 This might be a PyTorch+PyInstaller packaging issue")
                    else:
                        print(f"⚠️ GPU test failed in script: {e}")
                    
                    # Fallback to CPU
                    device = "cpu"
                    print(f"🖥️ Falling back to CPU mode: {device}")
                    return device
            else:
                device = "cpu"
                print(f"�️ No CUDA available, using CPU: {device}")
            
            return device
            
        except Exception as e:
            print(f"⚠️ Device detection error: {e}")
            print("🖥️ Emergency fallback to CPU mode")
            return "cpu"
    
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
        PyInstaller-safe version.
        
        Returns:
            dict: Device information
        """
        info = {
            "device": self.device,
            "device_type": "GPU" if self.device.startswith("cuda") else "CPU"
        }
        
        if self.device.startswith("cuda"):
            try:
                # Safer GPU info extraction for executables
                gpu_id = int(self.device.split(":")[1]) if ":" in self.device else 0
                
                # Try to get GPU info, but handle PyInstaller issues gracefully
                try:
                    info["gpu_name"] = torch.cuda.get_device_name(gpu_id)
                except Exception as e:
                    print(f"Warning: Could not get GPU name: {e}")
                    info["gpu_name"] = f"CUDA Device {gpu_id}"
                
                try:
                    info["gpu_memory"] = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                except Exception as e:
                    print(f"Warning: Could not get GPU memory: {e}")
                    info["gpu_memory"] = 0.0
                
                try:
                    info["cuda_version"] = torch.version.cuda
                except Exception:
                    info["cuda_version"] = "Unknown"
                    
            except Exception as e:
                print(f"Warning: GPU info collection failed: {e}")
                # Fallback info
                info["gpu_name"] = "CUDA Device"
                info["gpu_memory"] = 0.0
                info["cuda_version"] = "Unknown"
        else:
            # CPU info is usually safe
            try:
                import os
                info["cpu_threads"] = torch.get_num_threads()
                info["cpu_cores"] = os.cpu_count() or 4
            except Exception as e:
                print(f"Warning: CPU info collection failed: {e}")
                info["cpu_threads"] = 4
                info["cpu_cores"] = 4
            
        return info
    
    def should_tile_image(self, gray8):
        """Check if image should be tiled based on size and configuration."""
        if not hasattr(self, 'enable_tiling'):
            # Fallback for backward compatibility
            return False
        if not self.enable_tiling:
            return False
        h, w = gray8.shape
        return max(h, w) > self.min_image_size_for_tiling

    def create_tiles(self, gray8):
        """Split image into overlapping tiles for better grain detection."""
        h, w = gray8.shape
        tiles = []
        tile_positions = []
        
        stride = self.tile_size - self.tile_overlap
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Calculate tile boundaries
                y1 = y
                y2 = min(y + self.tile_size, h)
                x1 = x
                x2 = min(x + self.tile_size, w)
                
                # Extract tile
                tile = gray8[y1:y2, x1:x2]
                
                # Pad tile if it's smaller than tile_size (edge tiles)
                if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
                    padded = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded
                
                tiles.append(tile)
                tile_positions.append((y1, y2, x1, x2))
        
        return tiles, tile_positions

    def deduplicate_masks(self, masks):
        """Remove duplicate/overlapping masks from tiling."""
        if masks.shape[0] == 0:
            return masks
        
        # Calculate IoU (Intersection over Union) for all pairs
        unique_masks = []
        used = set()
        
        for i in range(masks.shape[0]):
            if i in used:
                continue
            
            mask_i = masks[i] > 0.5
            if not mask_i.any():
                continue
            
            # Keep this mask
            unique_masks.append(masks[i])
            used.add(i)
            
            # Find and mark duplicates
            for j in range(i + 1, masks.shape[0]):
                if j in used:
                    continue
                
                mask_j = masks[j] > 0.5
                if not mask_j.any():
                    continue
                
                # Calculate IoU
                intersection = np.logical_and(mask_i, mask_j).sum()
                union = np.logical_or(mask_i, mask_j).sum()
                
                if union > 0:
                    iou = intersection / union
                    # If IoU > 60%, consider it a duplicate
                    if iou > 0.6:
                        used.add(j)
        
        if len(unique_masks) > 0:
            return np.array(unique_masks)
        else:
            return np.array([]).reshape(0, masks.shape[1], masks.shape[2])

    def edges_from_variant(self, im_float01, tv_weight=0.01, ridge_percentile=70, min_size=50):
        """
        Auto-detect bright or dark grain boundaries using Sato ridge detection.
        Compares both modes and uses the stronger one.
        
        Args:
            im_float01: Input image normalized to [0,1] (grayscale or RGB)
            tv_weight: Total variation denoising weight
            ridge_percentile: Percentile for threshold calculation
            min_size: Minimum object size to keep
            
        Returns:
            Binary edge mask (boolean array)
        """
        if not RIDGE_FILTERING_AVAILABLE:
            raise RuntimeError("Ridge filtering requires scikit-image. Install with: pip install scikit-image>=0.19.0")
        
        # Convert RGB to grayscale if needed
        if len(im_float01.shape) == 3:
            # RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
            im_gray = 0.299 * im_float01[:,:,0] + 0.587 * im_float01[:,:,1] + 0.114 * im_float01[:,:,2]
        else:
            im_gray = im_float01
        
        # Denoise image
        dn = denoise_tv_chambolle(im_gray, weight=tv_weight, max_num_iter=50)
        
        # Try both dark and bright ridges
        ridge_dark = sato(dn, sigmas=range(1, 5), black_ridges=True)
        ridge_bright = sato(dn, sigmas=range(1, 5), black_ridges=False)
        
        # Compare strength at percentile
        strength_dark = np.percentile(ridge_dark, ridge_percentile)
        strength_bright = np.percentile(ridge_bright, ridge_percentile)
        
        # Use stronger response
        if strength_dark > strength_bright:
            print(f"  Auto-detected DARK boundaries (strength: {strength_dark:.4f} vs {strength_bright:.4f})")
            ridge = ridge_dark
        else:
            print(f"  Auto-detected BRIGHT boundaries (strength: {strength_bright:.4f} vs {strength_dark:.4f})")
            ridge = ridge_bright
        
        # Threshold and skeletonize
        thresh = np.percentile(ridge, ridge_percentile)
        edges = ridge > thresh
        edges = skeletonize(edges)
        
        # Remove small objects
        edges = remove_small_objects(edges.astype(bool), min_size=min_size)
        
        # Return both edges (binary) and ridge (float response) - like experimental UI
        return edges, ridge
    
    def color_sam_masks_by_ridge(self, rgb_base, ridge, m_np, ridge_threshold=0.15):
        """
        Color SAM masks based on ridge content with anti-overwrite protection.
        Each accepted grain gets a unique color.
        
        Args:
            rgb_base: Base RGB image to draw on
            ridge: Ridge detection result (float array)
            m_np: SAM masks as numpy array
            ridge_threshold: Minimum ridge density to accept grain
            
        Returns:
            tuple: (colored_image, accepted_masks_list, grain_stats_dict)
        """
        if not RIDGE_FILTERING_AVAILABLE:
            raise RuntimeError("Ridge filtering requires scikit-image")
        
        colored = rgb_base.copy()
        colored_mask = np.zeros(rgb_base.shape[:2], dtype=bool)  # Track colored pixels
        accepted_masks = []
        grain_stats = {}
        
        # Normalize ridge to 0-1 range (like experimental UI)
        r_min, r_max = float(ridge.min()), float(ridge.max())
        if r_max > r_min:
            ridge_norm = (ridge - r_min) / (r_max - r_min)
        else:
            ridge_norm = np.zeros_like(ridge, dtype=np.float32)
        
        kernel = np.ones((3, 3), np.uint8)
        total_masks = len(m_np)
        
        for i, m_bool in enumerate(m_np):
            if not m_bool.any():
                continue
            
            # Calculate ridge density on BOUNDARY (like experimental UI)
            mask_u8 = (m_bool.astype(np.uint8) * 255)
            eroded = cv2.erode(mask_u8, kernel, iterations=1)
            boundary = (mask_u8 ^ eroded).astype(bool)
            
            if not boundary.any():
                continue
            
            # Mean ridge on boundary
            mean_ridge = ridge_norm[boundary].mean()
            
            if mean_ridge >= ridge_threshold:
                # Generate unique color for this grain
                np.random.seed(i)
                color = tuple(np.random.randint(50, 256, size=3).tolist())
                
                # Only color pixels that haven't been colored yet (anti-overwrite)
                new_pixels = m_bool & ~colored_mask
                
                if new_pixels.sum() > 0:
                    colored[new_pixels] = color
                    colored_mask[new_pixels] = True
                    accepted_masks.append(m_bool)
                    
                    grain_stats[len(accepted_masks)] = {
                        'original_index': i,
                        'mean_ridge': float(mean_ridge),
                        'pixels': int(m_bool.sum()),
                        'new_pixels': int(new_pixels.sum()),
                        'color': color
                    }
        
        print(f"  Ridge filtering: {len(accepted_masks)}/{total_masks} masks accepted (threshold={ridge_threshold})")
        
        return colored, accepted_masks, grain_stats
    
    def count_unique_grains_by_color(self, colored_image, background_color=(0, 0, 0)):
        """
        Count unique grains by counting unique RGB colors.
        Each grain has a unique color, so unique colors = grain count.
        
        Args:
            colored_image: RGB image with uniquely colored grains
            background_color: Background color to exclude from count
            
        Returns:
            int: Number of unique grains (unique colors)
        """
        # Reshape to list of RGB tuples
        h, w = colored_image.shape[:2]
        pixels = colored_image.reshape(h * w, 3)
        
        # Get unique colors
        unique_colors = np.unique(pixels, axis=0)
        
        # Remove background color
        bg = np.array(background_color)
        unique_colors = unique_colors[~np.all(unique_colors == bg, axis=1)]
        
        return len(unique_colors)
    
    def segment_grains(self, rgb_image, verbose=False, apply_ridge_filtering=False, 
                       ridge_config=None, enable_tiling=None):
        """
        Segment grains using SAM model with optional ridge-based filtering and tiling.
        
        Args:
            rgb_image: RGB image (uint8, HxWx3)
            verbose: Whether to print verbose output
            apply_ridge_filtering: Whether to apply ridge-based filtering
            ridge_config: Dictionary with ridge filtering parameters:
                - ridge_threshold: Minimum ridge density (default: 0.15)
                - tv_weight: Total variation denoising weight (default: 0.01)
                - ridge_percentile: Percentile for threshold (default: 70)
                - min_size: Minimum edge object size (default: 50)
            enable_tiling: Override tiling setting (None = use config)
            
        Returns:
            tuple: (segmentation_results, processing_time, filtered_data)
                   filtered_data is None if ridge filtering not applied,
                   otherwise dict with: colored_image, accepted_masks, grain_stats, unique_grain_count
        """
        start_time = time.time()
        
        # Convert RGB to grayscale for tiling check
        gray8 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # Check if we should use tiling
        use_tiling = enable_tiling if enable_tiling is not None else (
            hasattr(self, 'enable_tiling') and self.enable_tiling and self.should_tile_image(gray8)
        )
        
        if use_tiling:
            h, w = gray8.shape
            print(f"🔍 Image is large ({max(h, w)}px), using tiled processing...")
            
            # Create tiles
            tiles, tile_positions = self.create_tiles(gray8)
            print(f"  Split into {len(tiles)} tiles of {self.tile_size}x{self.tile_size} px with {self.tile_overlap} px overlap")
            
            # Process each tile
            all_masks = []
            for tile_idx, (tile, (y1, y2, x1, x2)) in enumerate(zip(tiles, tile_positions)):
                if verbose:
                    print(f"  Processing tile {tile_idx+1}/{len(tiles)}: position ({y1}:{y2}, {x1}:{x2})")
                
                # Convert tile to RGB
                tile_rgb = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
                
                # Run SAM on tile
                tile_results = self.model.predict(
                    tile_rgb, 
                    device=self.device, 
                    verbose=False,
                    half=self.device.startswith("cuda")
                )
                
                # Extract masks from tile
                if tile_results and len(tile_results) > 0 and tile_results[0].masks is not None:
                    tile_masks_data = tile_results[0].masks.data
                    if hasattr(tile_masks_data, 'cpu'):
                        tile_masks = tile_masks_data.cpu().numpy()
                    else:
                        tile_masks = np.array(tile_masks_data)
                    
                    # Offset masks to full image coordinates
                    if tile_masks.ndim == 3:
                        for mask in tile_masks:
                            # Crop to actual tile size (remove padding)
                            actual_h = y2 - y1
                            actual_w = x2 - x1
                            mask_cropped = mask[:actual_h, :actual_w]
                            
                            # Create full-size mask
                            full_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=mask.dtype)
                            full_mask[y1:y2, x1:x2] = mask_cropped
                            all_masks.append(full_mask)
            
            # Combine all tile masks
            if all_masks:
                combined_masks = np.array(all_masks)
                print(f"  Collected {len(all_masks)} masks from all tiles")
                
                # Deduplicate overlapping masks
                print(f"  Deduplicating masks from tile overlaps...")
                combined_masks = self.deduplicate_masks(combined_masks)
                print(f"  ✅ After deduplication: {combined_masks.shape[0]} unique masks")
                
                # Create mock result object
                class MockMasks:
                    def __init__(self, data):
                        self.data = data
                
                class MockResult:
                    def __init__(self, masks):
                        self.masks = masks
                
                results = [MockResult(MockMasks(combined_masks))]
            else:
                results = []
        else:
            # Original single-image processing
            if verbose and hasattr(self, 'enable_tiling'):
                h, w = gray8.shape
                print(f"  Image size {max(h, w)}px is below tiling threshold ({self.min_image_size_for_tiling}px)")
                print("  Processing as single image...")
            
            results = self.model.predict(
                rgb_image, 
                device=self.device, 
                verbose=verbose,
                half=self.device.startswith("cuda")
            )
        
        processing_time = time.time() - start_time
        
        # Apply ridge filtering if requested
        filtered_data = None
        if apply_ridge_filtering and RIDGE_FILTERING_AVAILABLE:
            try:
                # Default ridge config
                config = {
                    'ridge_threshold': 0.15,
                    'tv_weight': 0.01,
                    'ridge_percentile': 70,
                    'min_size': 50
                }
                if ridge_config:
                    config.update(ridge_config)
                
                print("\n🔬 Applying ridge-based filtering...")
                
                # Convert image to float [0,1]
                im_float01 = rgb_image.astype(np.float32) / 255.0
                
                # Detect edges (returns both edges and ridge response)
                print("  Detecting grain boundaries...")
                edges, ridge = self.edges_from_variant(
                    im_float01,
                    tv_weight=config['tv_weight'],
                    ridge_percentile=config['ridge_percentile'],
                    min_size=config['min_size']
                )
                
                # Extract masks from results
                if results and len(results) > 0 and results[0].masks is not None:
                    masks_data = results[0].masks.data
                    if hasattr(masks_data, 'cpu'):
                        m_np = masks_data.cpu().numpy().astype(bool)
                    else:
                        m_np = np.array(masks_data, dtype=bool)
                    
                    # Apply ridge filtering using float ridge response
                    print("  Filtering masks by ridge content...")
                    colored_image, accepted_masks, grain_stats = self.color_sam_masks_by_ridge(
                        rgb_image.copy(),
                        ridge,
                        m_np,
                        ridge_threshold=config['ridge_threshold']
                    )
                    
                    # Count unique grains
                    unique_grain_count = self.count_unique_grains_by_color(colored_image)
                    print(f"  ✅ Unique grains detected: {unique_grain_count}")
                    
                    filtered_data = {
                        'colored_image': colored_image,
                        'accepted_masks': accepted_masks,
                        'grain_stats': grain_stats,
                        'unique_grain_count': unique_grain_count,
                        'edges': edges
                    }
                else:
                    print("  ⚠️ No masks to filter")
                    
            except Exception as e:
                print(f"  ⚠️ Ridge filtering failed: {e}")
                print("  Continuing with unfiltered results")
        
        # Clean up GPU memory after processing
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            
        return results, processing_time, filtered_data
    
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

    def create_label_from_masks(self, mask_list, min_area=0):
        """
        Convert a list of boolean masks to a label image.
        Used for ridge-filtered masks.
        
        Args:
            mask_list: List of boolean numpy arrays (masks)
            min_area: Minimum area threshold in pixels
            
        Returns:
            Label image (numpy array) or None if no valid masks
        """
        if not mask_list:
            return None
        
        # Get dimensions from first mask
        h, w = mask_list[0].shape
        label_img = np.zeros((h, w), dtype=np.int32)
        
        idx = 1
        for mask_bool in mask_list:
            if min_area > 0 and mask_bool.sum() < min_area:
                continue
            label_img[mask_bool] = idx
            idx += 1
        
        if idx == 1:  # No masks added
            return None
            
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


def create_complete_analyzer(model_gpu="sam_b.pt", model_cpu="sam_b.pt", 
                           device=None, um_per_pixel=1.0,
                           enable_tiling=True, tile_size=1024, tile_overlap=128, 
                           min_image_size_for_tiling=2048):
    """
    Create a complete grain analysis setup.
    
    Args:
        model_gpu: SAM model for GPU (default: sam_b.pt for precision)
        model_cpu: SAM model for CPU (default: sam_b.pt for precision)
        device: Device to use
        um_per_pixel: Pixel to micron conversion
        enable_tiling: Enable automatic tiling for large images
        tile_size: Size of each tile in pixels
        tile_overlap: Overlap between tiles to avoid edge artifacts
        min_image_size_for_tiling: Only tile images larger than this
        
    Returns:
        tuple: (GrainAnalyzer, FeretCalculator, GrainMeasurements)
    """
    analyzer = GrainAnalyzer(model_gpu, model_cpu, device,
                            enable_tiling=enable_tiling, 
                            tile_size=tile_size, 
                            tile_overlap=tile_overlap,
                            min_image_size_for_tiling=min_image_size_for_tiling)
    feret_calc = FeretCalculator()
    measurements = GrainMeasurements(um_per_pixel)
    
    return analyzer, feret_calc, measurements