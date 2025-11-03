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

# Set Tesseract path (Windows installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class ExactFooterOCR:
    """Exact OCR processor for SEM image footers."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
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
            
            # Debug: Print image info
            print(f"🔍 Debug - Loaded image shape: {img.shape}, dtype: {img.dtype}")
            
            height, width = img.shape[:2]
            
            # Extract larger bottom footer region (15% instead of 5% to capture more text)
            footer_height = max(int(height * footer_height_ratio), 100)  # Minimum 100px
            footer_region = img[-footer_height:, :]
            
            print(f"🔍 Debug - Footer region shape: {footer_region.shape}")
            print(f"🔍 Debug - Footer height: {footer_height}px ({footer_height_ratio*100:.1f}% of image)")
            
            return footer_region, (width, height)
            
        except Exception as e:
            print(f"❌ Error in extract_footer_region: {e}")
            raise

    def preprocess_footer_for_ocr(self, footer_img):
        """Advanced preprocessing for SEM footer OCR with improved handling."""
        # Ensure we have the right image format
        if footer_img is None:
            raise ValueError("Footer image is None")
        
        print(f"🔍 Debug - Processing image shape: {footer_img.shape}")
        
        # Convert to grayscale if needed
        if len(footer_img.shape) == 3:
            # Check if it's BGR (OpenCV default) or RGB
            if footer_img.shape[2] == 3:
                gray = cv2.cvtColor(footer_img, cv2.COLOR_BGR2GRAY)
            elif footer_img.shape[2] == 4:
                gray = cv2.cvtColor(footer_img, cv2.COLOR_BGRA2GRAY)
            elif footer_img.shape[2] == 1:
                # Single channel image - squeeze the last dimension
                gray = footer_img.squeeze(axis=2)
            else:
                raise ValueError(f"Unexpected number of channels: {footer_img.shape[2]}")
        elif len(footer_img.shape) == 2:
            # Already grayscale
            gray = footer_img.copy()
        else:
            raise ValueError(f"Unexpected image shape: {footer_img.shape}")
        
        print(f"🔍 Debug - Grayscale image shape: {gray.shape}")
        
        processed_images = []
        
        # Method 1: Enhanced contrast for white text on dark background
        enhanced = cv2.convertScaleAbs(gray, alpha=2.5, beta=30)
        processed_images.append(enhanced)
        
        # Method 2: Invert if the footer has light text on dark background
        # Check if image is predominantly dark (SEM footers often are)
        mean_intensity = np.mean(gray)
        if mean_intensity < 128:  # Dark background
            inverted = cv2.bitwise_not(gray)
            processed_images.append(inverted)
        
        # Method 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(gray)
        processed_images.append(clahe_img)
        
        # Method 4: Apply adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        processed_images.append(adaptive_thresh)
        
        # Method 5: Apply OTSU threshold
        _, otsu_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(otsu_thresh)
        
        # Method 6: Morphological operations to clean up
        kernel = np.ones((2,2), np.uint8)
        morphed = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)
        processed_images.append(morphed)
        
        # Method 7: Gaussian blur + threshold for noise reduction
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, blur_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(blur_thresh)
        
        # Method 8: Upscale for better OCR (small text)
        height, width = gray.shape
        if height < 200:  # If footer is small, upscale
            scale_factor = 3
            upscaled = cv2.resize(enhanced, (width * scale_factor, height * scale_factor), 
                                interpolation=cv2.INTER_CUBIC)
            processed_images.append(upscaled)
        
        return processed_images

    def perform_ocr_with_multiple_methods(self, footer_img):
        """Perform OCR with multiple preprocessing methods."""
        all_results = []
        
        # Different preprocessing methods
        processed_imgs = self.preprocess_footer_for_ocr(footer_img)
        
        # Different OCR configurations - improved for SEM footer text
        ocr_configs = [
            # Page segmentation modes optimized for SEM footers
            '--psm 6 -c tessedit_char_whitelist=0123456789.,μumkVPaFWfwBSDSEETDxX:- ',  # Single uniform block
            '--psm 7 -c tessedit_char_whitelist=0123456789.,μumkVPaFWfwBSDSEETDxX:- ',  # Single text line
            '--psm 8 -c tessedit_char_whitelist=0123456789.,μumkVPaFWfwBSDSEETDxX:- ',  # Single word
            '--psm 13 -c tessedit_char_whitelist=0123456789.,μumkVPaFWfwBSDSEETDxX:- ', # Raw line (no formatting)
            
            # Without character whitelist (in case special chars are causing issues)
            '--psm 6',
            '--psm 7', 
            '--psm 8',
            '--psm 13',
            
            # Legacy engine for different text styles
            '--psm 6 --oem 0',
            '--psm 7 --oem 0',
            
            # Sparse text mode for scattered elements
            '--psm 11',
            '--psm 12',
            
            # Default auto mode
            '--psm 3',
        ]
        
        for i, processed_img in enumerate(processed_imgs):
            for j, config in enumerate(ocr_configs):
                try:
                    text = pytesseract.image_to_string(processed_img, config=config).strip()
                    if text:
                        all_results.append({
                            'text': text,
                            'preprocessing': f'method_{i}',
                            'ocr_config': f'config_{j}',
                            'confidence': len(text)
                        })
                except Exception as e:
                    continue
        
        return all_results

    def parse_sem_metadata_exact(self, text_results):
        """Parse exact SEM metadata from OCR results."""
        metadata = {}
        
        if not text_results:
            return metadata
        
        # Combine all text results for analysis
        all_text = " ".join([result['text'] for result in text_results])
        best_result = text_results[0]['text'] if text_results else ""
        
        # Precise regex patterns for SEM footer data - improved for Frame Width detection
        patterns = {
            'fw_um': [
                # Primary Frame Width patterns
                r'FW[\s:]*([0-9]+\.?[0-9]*)\s*[μu]?m',           # FW 21.8 μm, FW: 21.8 μm
                r'Fw[\s:]*([0-9]+\.?[0-9]*)\s*[μu]?m',           # Fw 21.8 μm, Fw: 21.8 μm  
                r'Frame[\s]*Width[\s:]*([0-9]+\.?[0-9]*)\s*[μu]?m', # Frame Width 21.8 μm
                r'frame[\s]*width[\s:]*([0-9]+\.?[0-9]*)\s*[μu]?m', # frame width 21.8 μm
                
                # Reverse patterns (value before label)
                r'([0-9]+\.?[0-9]*)\s*[μu]?m[\s]*FW',            # 21.8 μm FW
                r'([0-9]+\.?[0-9]*)\s*[μu]?m[\s]*Fw',            # 21.8 μm Fw
                r'([0-9]+\.?[0-9]*)\s*[μu]?m[\s]*Frame[\s]*Width', # 21.8 μm Frame Width
                
                # Without units (assuming micrometers)
                r'FW[\s:]*([0-9]+\.?[0-9]*)',                    # FW 21.8
                r'Fw[\s:]*([0-9]+\.?[0-9]*)',                    # Fw 21.8
                
                # Alternative unit spellings
                r'FW[\s:]*([0-9]+\.?[0-9]*)\s*um',               # FW 21.8 um
                r'Fw[\s:]*([0-9]+\.?[0-9]*)\s*um',               # Fw 21.8 um
                r'([0-9]+\.?[0-9]*)\s*um[\s]*[Ff][Ww]',          # 21.8 um FW
                
                # With magnification context (common pattern)
                r'([0-9]+\.?[0-9]*)\s*[μu]m\s+[0-9]+[kK][Vv]',  # 21.8 μm 10kV (pattern)
                r'x\s+([0-9]+\.?[0-9]*)\s*[μu]m',                # x 21.8 μm (magnification context)
                r'[0-9]+\s*x\s+([0-9]+\.?[0-9]*)\s*[μu]m',       # 500 x 21.8 μm
                
                # Edge cases with spacing/formatting issues
                r'F\s*W[\s:]*([0-9]+\.?[0-9]*)\s*[μu]?m?',       # F W 21.8 μm (spaced)
                r'([0-9]+\.?[0-9]*)\s*μm\s*$',                   # 21.8 μm at end of line
                r'([0-9]+\.?[0-9]*)\s*um\s*$',                   # 21.8 um at end of line
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
            for pattern in field_patterns:
                matches = re.findall(pattern, best_result, re.IGNORECASE)
                if matches:
                    try:
                        # Clean up the matched value
                        value_str = matches[0].replace(',', '').replace(' ', '')
                        if field == 'magnification':
                            metadata[field] = int(float(value_str))
                        else:
                            metadata[field] = float(value_str)
                        break  # Use first successful match
                    except (ValueError, IndexError):
                        continue
        
        return metadata

    def analyze_sem_footer_exact(self, image_path):
        """
        Main method to analyze SEM footer and extract exact metadata.
        Returns dictionary with metadata and calculated pixel size.
        """
        try:
            print(f"🔍 Starting OCR analysis for: {image_path}")
            
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
            
            print(f"✅ OCR analysis completed successfully")
            
            return {
                'image_path': image_path,
                'image_size': image_size,
                'metadata': metadata,
                'um_per_pixel': um_per_pixel,
                'ocr_text': ocr_results[0]['text'] if ocr_results else '',
                'success': 'fw_um' in metadata
            }
            
        except Exception as e:
            error_msg = f"Error analyzing SEM footer: {e}"
            print(f"❌ {error_msg}")
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
            print(f"🔍 Extracting frame width from: {os.path.basename(image_path)}")
            
            # Use the comprehensive analysis method
            results = self.analyze_sem_footer_exact(image_path)
            
            if 'error' in results:
                print(f"❌ Error in frame width extraction: {results['error']}")
                return None
            
            metadata = results.get('metadata', {})
            frame_width = metadata.get('fw_um')
            
            if frame_width:
                print(f"✅ Frame width detected: {frame_width} μm")
                return frame_width
            else:
                print("⚠️ Frame width not found in image footer")
                
                # Debug: Print what was extracted
                if 'ocr_text' in results:
                    print(f"🔍 OCR Text found: {results['ocr_text'][:200]}...")
                
                return None
                
        except Exception as e:
            print(f"❌ Exception in extract_frame_width: {e}")
            import traceback
            traceback.print_exc()
            return None


def create_exact_footer_ocr():
    """Factory function to create exact footer OCR processor."""
    return ExactFooterOCR()