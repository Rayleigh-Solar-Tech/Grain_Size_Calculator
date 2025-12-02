#!/usr/bin/env python3
"""
Exact SEM Footer OCR Module for Grain Size Calculator
Integrates exact OCR functionality into the main application.
"""

import cv2
import numpy as np
import re
import pytesseract
from PIL import Image
import os
import logging
import signal
from functools import wraps
import sys

# Set Tesseract path - try bundled version first, then system paths
def setup_tesseract_path():
    """Setup Tesseract executable path - try bundled first, then system installations."""
    
    # If running as executable (PyInstaller), try bundled tesseract first
    if getattr(sys, 'frozen', False):
        # Running as executable
        bundle_dir = sys._MEIPASS
        bundled_tesseract = os.path.join(bundle_dir, 'tesseract', 'tesseract.exe')
        if os.path.exists(bundled_tesseract):
            print(f"Using bundled Tesseract: {bundled_tesseract}")
            return bundled_tesseract
    else:
        # Running as script, try relative path first
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        bundled_tesseract = os.path.join(script_dir, 'tesseract', 'tesseract.exe')
        if os.path.exists(bundled_tesseract):
            print(f"Using bundled Tesseract: {bundled_tesseract}")
            return bundled_tesseract
    
    # Try common system installation paths
    common_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe',
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            print(f"Using system Tesseract: {path}")
            return path
    
    # If nothing found, let pytesseract try its default
    print("Warning: Tesseract not found in common locations. Trying default...")
    return None

# Setup Tesseract path
tesseract_path = setup_tesseract_path()
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Timeout exception
class TimeoutError(Exception):
    pass

def timeout_handler(func, timeout_duration=10):
    """Wrapper to add timeout to a function (Windows compatible)."""
    def wrapper(*args, **kwargs):
        import threading
        result = [TimeoutError('OCR processing timeout')]
        
        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                result[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout_duration)
        
        if thread.is_alive():
            # Thread is still running - timeout occurred
            return {'error': f'OCR timeout after {timeout_duration} seconds'}
        
        # Check if result is an exception
        if isinstance(result[0], Exception):
            raise result[0]
        
        return result[0]
    
    return wrapper

class ExactFooterOCR:
    """Exact OCR processor for SEM image footers."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def quick_validate_sem_image(self, image_path):
        """
        Quick validation to detect obviously bad/non-SEM images BEFORE attempting OCR.
        Returns (is_valid, error_message)
        """
        try:
            # Check file exists
            if not os.path.exists(image_path):
                return False, "File does not exist"
            
            # Check file size (more lenient - SEM images can vary in size)
            file_size = os.path.getsize(image_path)
            if file_size < 10000:  # Less than 10KB - definitely too small
                return False, "File too small - not a SEM image"
            
            # Try to read image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False, "Cannot read image file"
            
            height, width = img.shape[:2]
            
            # More lenient size check - SEM images can be smaller sometimes
            if width < 256 or height < 256:
                return False, f"Image too small ({width}x{height}) - SEM images should be at least 256x256"
            
            # Check if image has reasonable contrast (not blank/uniform) - more lenient
            mean_intensity = np.mean(img)
            std_intensity = np.std(img)
            
            if std_intensity < 2:  # Very lenient - almost no variation
                return False, "Image appears blank or uniform - no content detected"
            
            # Check footer region exists and has some content - more lenient
            footer_height = max(int(height * 0.15), 50)  # Smaller minimum footer
            footer_region = img[-footer_height:, :]
            
            footer_std = np.std(footer_region)
            if footer_std < 1:  # Much more lenient footer check
                print(f"DEBUG: Footer std = {footer_std} (very low, but allowing through)")
            
            # More lenient text pattern check
            edges = cv2.Canny(footer_region, 30, 100)  # Lower thresholds
            edge_density = np.sum(edges > 0) / edges.size
            
            print(f"DEBUG: Edge density = {edge_density:.4f}")  # Debug output
            
            if edge_density < 0.003:  # Much more lenient - 0.3% instead of 1%
                print(f"DEBUG: Very low edge density ({edge_density:.4f}), but trying OCR anyway")
            
            return True, "Image passed validation"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
        
    def extract_footer_region(self, image_path, footer_height_ratio=0.15):
        """Extract the footer region from SEM image with larger region."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                # Try with different flags
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if img is None:
                    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        raise ValueError(f"Could not read image: {image_path}")
            
            height, width = img.shape[:2]
            print(f"DEBUG: Image dimensions: {width}x{height}")
            
            # Extract larger bottom footer region (15% instead of 5% to capture more text)
            footer_height = max(int(height * footer_height_ratio), 100)  # Minimum 100px
            print(f"DEBUG: Footer height calculated: {footer_height} pixels ({footer_height_ratio*100:.1f}% of {height})")
            
            footer_region = img[-footer_height:, :]
            print(f"DEBUG: Footer region extracted: {footer_region.shape}")
            
            return footer_region, (width, height)
            
        except Exception as e:
            print(f"❌ Error in extract_footer_region: {e}")
            raise

    def preprocess_footer_for_ocr(self, footer_img):
        """Optimized preprocessing for SEM footer OCR - only essential methods."""
        # Ensure we have the right image format
        if footer_img is None:
            raise ValueError("Footer image is None")
        
        # Convert to grayscale if needed
        if len(footer_img.shape) == 3:
            if footer_img.shape[2] == 3:
                gray = cv2.cvtColor(footer_img, cv2.COLOR_BGR2GRAY)
            elif footer_img.shape[2] == 4:
                gray = cv2.cvtColor(footer_img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = footer_img.squeeze(axis=2)
        else:
            gray = footer_img.copy()
        
        processed_images = []
        
        # Method 1: Enhanced contrast (works for most SEM footers)
        enhanced = cv2.convertScaleAbs(gray, alpha=2.5, beta=30)
        processed_images.append(enhanced)
        
        # Method 2: Invert for dark backgrounds (common in SEM)
        mean_intensity = np.mean(gray)
        if mean_intensity < 128:
            inverted = cv2.bitwise_not(enhanced)
            processed_images.append(inverted)
        
        # Method 3: OTSU threshold (reliable for text extraction)
        _, otsu_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(otsu_thresh)
        
        return processed_images

    def perform_ocr_with_multiple_methods(self, footer_img, max_attempts=12):
        """Optimized OCR with only the most effective configurations and early exit."""
        all_results = []
        attempts = 0
        
        print(f"DEBUG: Starting OCR with footer image shape: {footer_img.shape}")
        
        # Get preprocessed images (now only 3 methods instead of 8)
        processed_imgs = self.preprocess_footer_for_ocr(footer_img)
        print(f"DEBUG: Got {len(processed_imgs)} preprocessed images")
        
        # Only the most effective OCR configurations (4 instead of 13)
        ocr_configs = [
            '--psm 6 -c tessedit_char_whitelist=0123456789.,μumkVPaFWfwBSDSEETDxX:- ',  # Single block with whitelist
            '--psm 7 -c tessedit_char_whitelist=0123456789.,μumkVPaFWfwBSDSEETDxX:- ',  # Single line with whitelist
            '--psm 6',  # Single block without whitelist
            '--psm 7',  # Single line without whitelist
        ]
        
        # Total OCR attempts: 3 preprocessing × 4 configs = 12 (down from 104!)
        # But we'll limit to max_attempts for bad images
        for i, processed_img in enumerate(processed_imgs):
            for j, config in enumerate(ocr_configs):
                # Early exit if we've tried enough times without finding anything
                attempts += 1
                print(f"DEBUG: OCR attempt {attempts}: preprocessing method {i}, config {j}")
                
                if attempts > max_attempts and not all_results:
                    # Tried several times with no results - likely not a SEM image
                    print(f"DEBUG: Reached max attempts ({max_attempts}) with no results")
                    return all_results
                
                try:
                    text = pytesseract.image_to_string(processed_img, config=config).strip()
                    print(f"DEBUG: OCR result (method {i}, config {j}): '{text[:100]}...' (length: {len(text)})")
                    
                    if text:
                        all_results.append({
                            'text': text,
                            'preprocessing': f'method_{i}',
                            'ocr_config': f'config_{j}',
                            'confidence': len(text)
                        })
                        # If we found text with Frame Width, stop early
                        if 'FW' in text or 'Fw' in text or 'μm' in text or 'um' in text:
                            print(f"DEBUG: Found Frame Width related text, stopping early")
                            return all_results  # Early exit when we find relevant text
                except Exception as e:
                    print(f"DEBUG: OCR error (method {i}, config {j}): {e}")
                    continue
        
        print(f"DEBUG: OCR completed, total results: {len(all_results)}")
        return all_results

    def parse_sem_metadata_exact(self, text_results):
        """Parse exact SEM metadata from OCR results."""
        metadata = {}
        
        if not text_results:
            print("DEBUG: No text results provided for parsing")
            return metadata
        
        # Combine all text results for analysis
        all_text = " ".join([result['text'] for result in text_results])
        best_result = text_results[0]['text'] if text_results else ""
        
        print(f"DEBUG: Parsing metadata from combined text: '{all_text[:200]}...'")
        print(f"DEBUG: Best result: '{best_result[:100]}...')")
        
        # Debug: Show what's around FW in the text
        import re
        fw_context = re.findall(r'.{0,10}FW.{0,10}', best_result, re.IGNORECASE)
        if fw_context:
            print(f"DEBUG: Found FW context: {fw_context}")
        else:
            print("DEBUG: No FW found in text")
        
        # Precise regex patterns for SEM footer data - improved for Frame Width detection
        patterns = {
            'fw_um': [
                # SIMPLE FW (Frame Width) patterns only - focus on what matters
                r'FW[\s:]*([0-9]+\.?[0-9]*)',                    # FW 19.9 or FW: 19.9
                r'Fw[\s:]*([0-9]+\.?[0-9]*)',                    # Fw 19.9 or Fw: 19.9
                r'F\s*W[\s:]*([0-9]+\.?[0-9]*)',                 # F W 19.9 (spaced)
                
                # Pattern for SEM footer context: magnification x Frame_Width voltage
                r'[0-9]+x\s*([0-9]+\.?[0-9]*)(?:um|μm|m)\s*[0-9]+[kK][Vv]',  # 4300x 120um 10kV - extract Frame Width
                r'[0-9]+\s*x\s*([0-9]+\.?[0-9]*)(?:um|μm|m)\s*[0-9]+[kK]',   # 4300 x 120um 10kV - extract Frame Width
            ],
            'magnification': [
                r'([0-9,\s]+)\s*[xX×]',                          # 23,500 x or 23500x
                r'Mag[\s]*([0-9,\s]+)',                          # Mag 23500
                r'M[\s]*([0-9,\s]+)\s*[xX×]',                    # M 23500 x
                r'([0-9]+[\s,]*[0-9]*)\s*[xX×]',                 # 23 500 x
                r'x\s*([0-9,\s]+)',                              # x 23500
            ],
            'voltage_kv': [
                r'HV[\s]*([0-9]+\.?[0-9]*)\s*[kK][Vv]',          # HV 10 kV
                r'([0-9]+\.?[0-9]*)\s*[kK][Vv]',                 # 10 kV
                r'([0-9]+)\s*k[Vv]',                             # 10 kV
                r'([0-9]+\.?[0-9]*)\s*KV',                       # 10 KV
            ],
            'working_distance_mm': [
                r'WD[\s]*([0-9]+\.?[0-9]*)\s*mm',                # WD 7.227 mm
                r'WD[\s]*([0-9]+\.?[0-9]*)',                     # WD 7.227
                r'([0-9]+\.?[0-9]*)\s*mm.*WD',                   # 7.227 mm WD
                r'([0-9]+\.[0-9]+)mm',                           # 7.227mm
                r'Working[\s]*Distance[\s]*([0-9]+\.?[0-9]*)',   # Working Distance 7.227
            ],
        }
        
        # Extract metadata using patterns
        for field, field_patterns in patterns.items():
            print(f"DEBUG: Looking for {field}...")
            for i, pattern in enumerate(field_patterns):
                matches = re.findall(pattern, best_result, re.IGNORECASE)
                if matches:
                    print(f"DEBUG: Pattern {i} for {field} matched: {matches}")
                    try:
                        # Clean up the matched value
                        value_str = matches[0].replace(',', '').replace(' ', '')
                        if field == 'magnification':
                            metadata[field] = int(float(value_str))
                        else:
                            metadata[field] = float(value_str)
                        print(f"DEBUG: Successfully extracted {field} = {metadata[field]}")
                        break  # Use first successful match
                    except (ValueError, IndexError) as e:
                        print(f"DEBUG: Failed to parse {field} value '{matches[0]}': {e}")
                        continue
                else:
                    print(f"DEBUG: No match for {field} pattern {i}: {pattern}")
            
            if field not in metadata:
                print(f"DEBUG: Failed to extract {field} from text")
        
        return metadata

    def _analyze_sem_footer_internal(self, image_path):
        """Internal method to analyze SEM footer (wrapped with timeout)."""
        # Extract footer region
        footer_region, image_size = self.extract_footer_region(image_path)
        
        # Perform OCR with multiple methods
        ocr_results = self.perform_ocr_with_multiple_methods(footer_region)
        
        if not ocr_results:
            return {'error': 'No OCR results obtained', 'image_size': image_size}
        
        # Parse metadata
        metadata = self.parse_sem_metadata_exact(ocr_results)
        
        # Calculate pixel size if frame width is available
        um_per_pixel = None
        if 'fw_um' in metadata and image_size:
            um_per_pixel = metadata['fw_um'] / image_size[0]  # Frame width / image width
        
        return {
            'image_path': image_path,
            'image_size': image_size,
            'metadata': metadata,
            'um_per_pixel': um_per_pixel,
            'ocr_text': ocr_results[0]['text'] if ocr_results else '',
            'success': 'fw_um' in metadata
        }
    
    def analyze_sem_footer_exact(self, image_path, timeout=10):
        """
        Main method to analyze SEM footer and extract exact metadata.
        Returns dictionary with metadata and calculated pixel size.
        
        Args:
            image_path: Path to the SEM image
            timeout: Maximum time in seconds to wait for OCR (default: 10s)
        """
        try:
            # STEP 1: Quick validation BEFORE attempting OCR (reject bad images fast!)
            is_valid, validation_msg = self.quick_validate_sem_image(image_path)
            if not is_valid:
                return {'error': f'Invalid SEM image: {validation_msg}'}
            
            # STEP 2: Wrap the internal method with timeout
            wrapped_func = timeout_handler(self._analyze_sem_footer_internal, timeout_duration=timeout)
            result = wrapped_func(image_path)
            return result
            
        except Exception as e:
            error_msg = f"Error analyzing SEM footer: {e}"
            self.logger.error(error_msg)
            import traceback
            traceback.print_exc()
            return {'error': error_msg}

    def extract_all_metadata(self, image_or_path):
        """
        Extract all metadata from image for compatibility with existing interface.
        This method matches the interface expected by the main application.
        """
        # Handle both image path and numpy array inputs
        if isinstance(image_or_path, str):
            image_path = image_or_path
        else:
            # If it's a numpy array, we need the original image path
            # For now, return empty metadata and let user input manually
            return {'frame_width_um': None, 'extracted_text': 'Array input not supported for footer OCR'}
        
        # Analyze the image
        results = self.analyze_sem_footer_exact(image_path)
        
        if 'error' in results:
            return {'frame_width_um': None, 'extracted_text': f"Error: {results['error']}"}
        
        # Format results for main application
        metadata = results.get('metadata', {})
        
        return {
            'frame_width_um': metadata.get('fw_um'),
            'magnification': metadata.get('magnification'),
            'voltage_kv': metadata.get('voltage_kv'),
            'working_distance_mm': metadata.get('working_distance_mm'),
            'um_per_pixel': results.get('um_per_pixel'),
            'extracted_text': results.get('ocr_text', ''),
            'full_metadata': metadata
        }

    def extract_frame_width(self, image_path):
        """
        Extract frame width from SEM image footer.
        This is the main method called by the GUI application.
        Returns frame width in micrometers or None if not found.
        """
        try:
            # Use the comprehensive analysis method
            results = self.analyze_sem_footer_exact(image_path)
            
            if 'error' in results:
                return None
            
            metadata = results.get('metadata', {})
            frame_width = metadata.get('fw_um')
            
            if frame_width:
                return frame_width
            else:
                return None
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None


def create_exact_footer_ocr():
    """Factory function to create exact footer OCR processor."""
    return ExactFooterOCR()