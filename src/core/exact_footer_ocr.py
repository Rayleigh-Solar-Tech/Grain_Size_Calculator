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
            
            height, width = img.shape[:2]
            
            # Extract larger bottom footer region (15% instead of 5% to capture more text)
            footer_height = max(int(height * footer_height_ratio), 100)  # Minimum 100px
            footer_region = img[-footer_height:, :]
            
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

    def perform_ocr_with_multiple_methods(self, footer_img):
        """Optimized OCR with only the most effective configurations."""
        all_results = []
        
        # Get preprocessed images (now only 3 methods instead of 8)
        processed_imgs = self.preprocess_footer_for_ocr(footer_img)
        
        # Only the most effective OCR configurations (4 instead of 13)
        ocr_configs = [
            '--psm 6 -c tessedit_char_whitelist=0123456789.,μumkVPaFWfwBSDSEETDxX:- ',  # Single block with whitelist
            '--psm 7 -c tessedit_char_whitelist=0123456789.,μumkVPaFWfwBSDSEETDxX:- ',  # Single line with whitelist
            '--psm 6',  # Single block without whitelist
            '--psm 7',  # Single line without whitelist
        ]
        
        # Total OCR attempts: 3 preprocessing × 4 configs = 12 (down from 104!)
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
                        # If we found text with Frame Width, stop early
                        if 'FW' in text or 'Fw' in text or 'μm' in text or 'um' in text:
                            return all_results  # Early exit when we find relevant text
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