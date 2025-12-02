"""
Image processing utilities for grain size analysis.
Handles image enhancement, preprocessing, and basic transformations.
"""

import cv2
import numpy as np
import math


def normalize01(array):
    """Normalize array to 0-1 range."""
    f = array.astype(np.float32)
    rng = float(f.max() - f.min()) or 1.0
    return (f - f.min()) / rng


def param_enhance(gray_img, clip=2.0, tile=8, gamma=1.0, unsharp_amount=1.15, unsharp_sigma=1.0):
    """
    Apply parametric enhancement to grayscale image.
    
    Args:
        gray_img: Input grayscale image
        clip: CLAHE clip limit
        tile: CLAHE tile grid size
        gamma: Gamma correction value
        unsharp_amount: Unsharp masking amount
        unsharp_sigma: Unsharp masking sigma
        
    Returns:
        Enhanced grayscale image
    """
    # Ensure uint8
    if gray_img.dtype != np.uint8:
        g8 = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        g8 = gray_img
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    g_clahe = clahe.apply(g8)
    
    # Apply gamma correction
    if gamma != 1.0:
        g_gamma = np.clip((g_clahe/255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
    else:
        g_gamma = g_clahe
    
    # Apply unsharp masking
    blur = cv2.GaussianBlur(g_gamma, (0, 0), unsharp_sigma)
    g_sharp = cv2.addWeighted(g_gamma, 1.0 + unsharp_amount, blur, -unsharp_amount, 0)
    
    return g_sharp


def prep_rgb8(gray_like):
    """Convert grayscale-like image to 8-bit RGB format for SAM input."""
    rgb = cv2.cvtColor(gray_like, cv2.COLOR_GRAY2RGB)
    
    if rgb.dtype != np.uint8:
        lo, hi = np.percentile(rgb, [0.5, 99.5])
        hi = max(hi, lo + 1e-6)
        rgb8 = np.clip((rgb - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb8 = rgb
        
    return np.ascontiguousarray(rgb8)


def load_and_convert_to_grayscale(img_path):
    """
    Load image and convert to grayscale if needed.
    
    Args:
        img_path: Path to image file
        
    Returns:
        tuple: (grayscale_image, original_height, original_width)
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"OpenCV could not read the image: {img_path}")
    
    # Convert to grayscale if needed
    if img.ndim == 3:
        if img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = img.mean(axis=2).astype(img.dtype)
    else:
        gray = img
    
    H, W = gray.shape[:2]
    return gray, H, W


def resize_for_processing(image, max_side, scale_factor=1.0):
    """
    Resize image for processing if needed (CPU optimization).
    
    Args:
        image: Input image
        max_side: Maximum allowed side length
        scale_factor: Current scale factor
        
    Returns:
        tuple: (resized_image, new_scale_factor)
    """
    H, W = image.shape[:2]
    long_side = max(H, W)
    
    if long_side > max_side:
        scale = max_side / long_side
        new_size = (int(W * scale), int(H * scale))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return resized, scale
    
    return image, scale_factor


def font_params_for_size(height, width):
    """Calculate appropriate font parameters based on image size."""
    diag = math.hypot(height, width)
    scale = max(0.35, min(0.9, diag / 3000.0))
    thickness = 1 if diag < 2500 else 2
    return scale, thickness


def create_overlay_visualization(base_image, label_image, grain_data, 
                               annotate_measurements=False, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Create visualization overlay with colored grains (experimental UI style).
    Each grain gets a unique random color, no length annotations by default.
    
    Args:
        base_image: Base grayscale image (normalized 0-1)
        label_image: Label image with grain IDs
        grain_data: List of grain measurement dictionaries
        annotate_measurements: Whether to annotate measurements (disabled by default)
        font: Font for text annotations
        
    Returns:
        RGB overlay image (0-1 range)
    """
    # Convert base image to RGB uint8 (0-255 range) for coloring
    base_u8 = (base_image * 255).astype(np.uint8)
    overlay_u8 = np.dstack([base_u8, base_u8, base_u8])
    
    # Color each grain with unique random color (like experimental UI)
    if label_image is not None and label_image.max() > 0:
        rng = np.random.default_rng(42)  # Fixed seed for consistency
        colored_mask = np.zeros(overlay_u8.shape[:2], dtype=bool)
        
        num_grains = int(label_image.max())
        for grain_id in range(1, num_grains + 1):
            mask = label_image == grain_id
            if not mask.any():
                continue
            
            # Generate unique color for this grain (80-255 range for visibility)
            color = rng.integers(low=80, high=255, size=3, dtype=np.uint8)
            
            # Only color pixels that haven't been colored yet (anti-overwrite)
            new_pixels = mask & ~colored_mask
            if new_pixels.any():
                overlay_u8[new_pixels, 0] = color[0]
                overlay_u8[new_pixels, 1] = color[1]
                overlay_u8[new_pixels, 2] = color[2]
                colored_mask |= new_pixels
    
    # Convert back to 0-1 range for consistency with rest of pipeline
    overlay = overlay_u8.astype(np.float32) / 255.0
    
    # Optionally add length annotations (disabled by default now)
    if annotate_measurements and grain_data:
        H, W = base_image.shape[:2]
        font_scale, font_thickness = font_params_for_size(H, W)
        
        for grain in grain_data:
            x1, y1 = int(round(grain["p1"][0])), int(round(grain["p1"][1]))
            x2, y2 = int(round(grain["p2"][0])), int(round(grain["p2"][1]))
            length_um = grain.get("length_um", 0)
            
            if length_um > 0:
                # Draw measurement line
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 1, 1), 1, cv2.LINE_AA)
                
                # Add text
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                txt = f"{length_um:.2f} µm"
                cv2.putText(overlay, txt, (mx + 1, my + 1), font, font_scale, 
                           (0, 0, 0), max(1, font_thickness + 1), cv2.LINE_AA)
                cv2.putText(overlay, txt, (mx, my), font, font_scale, 
                           (1, 1, 1), font_thickness, cv2.LINE_AA)
    
    return overlay