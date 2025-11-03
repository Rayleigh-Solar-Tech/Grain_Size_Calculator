"""
OCR module for extracting metadata from SEM images.
Handles extraction of scale information, frame width, and other parameters.
"""

import re
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. OCR functionality will be limited.")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class SEMImageOCR:
    """OCR for extracting metadata from SEM images."""
    
    def __init__(self, ocr_engine='easyocr', tesseract_path=None):
        """
        Initialize OCR engine.
        
        Args:
            ocr_engine: 'tesseract' or 'easyocr'
            tesseract_path: Path to tesseract executable (if needed)
        """
        self.ocr_engine = ocr_engine
        
        if ocr_engine == 'tesseract' and not TESSERACT_AVAILABLE:
            if EASYOCR_AVAILABLE:
                print("Tesseract not available, falling back to EasyOCR")
                self.ocr_engine = 'easyocr'
            else:
                raise ImportError("Neither tesseract nor easyocr available")
        
        if ocr_engine == 'easyocr' and not EASYOCR_AVAILABLE:
            if TESSERACT_AVAILABLE:
                print("EasyOCR not available, falling back to Tesseract")
                self.ocr_engine = 'tesseract'
            else:
                raise ImportError("Neither tesseract nor easyocr available")
        
        # Initialize OCR engines
        if self.ocr_engine == 'tesseract' and tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        if self.ocr_engine == 'easyocr':
            self.easyocr_reader = easyocr.Reader(['en'])
        
        # Patterns for extracting measurements
        self.scale_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:μm|um|µm|microns?)',  # Micron measurements
            r'(\d+(?:\.\d+)?)\s*(?:mm|millimeters?)',     # Millimeter measurements
            r'(\d+(?:\.\d+)?)\s*(?:nm|nanometers?)',      # Nanometer measurements
            r'Scale\s*[:|=]\s*(\d+(?:\.\d+)?)',           # Scale: value
            r'Frame\s*[wW]idth\s*[:|=]\s*(\d+(?:\.\d+)?)', # Frame width
        ]
        
        self.magnification_patterns = [
            r'(\d+(?:,\d+)?)\s*[×x]\s*(?:mag|magnification)?',
            r'[Mm]ag\s*[:|=]\s*(\d+(?:,\d+)?)',
            r'(\d+(?:,\d+)?)\s*[Xx]',
        ]
    
    def extract_text_from_image(self, image, region=None):
        """
        Extract text from image using OCR.
        
        Args:
            image: Input image (grayscale or RGB)
            region: Optional region (x, y, w, h) to focus OCR on
            
        Returns:
            Extracted text string
        """
        if region:
            x, y, w, h = region
            image = image[y:y+h, x:x+w]
        
        # Preprocess image for better OCR
        processed_img = self._preprocess_for_ocr(image)
        
        try:
            if self.ocr_engine == 'tesseract':
                text = pytesseract.image_to_string(processed_img, config='--psm 6')
            elif self.ocr_engine == 'easyocr':
                results = self.easyocr_reader.readtext(processed_img)
                text = ' '.join([result[1] for result in results])
            else:
                text = ""
        except Exception as e:
            logging.warning(f"OCR failed: {e}")
            text = ""
        
        return text.strip()
    
    def _preprocess_for_ocr(self, image):
        """Preprocess image for better OCR results."""
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Threshold to binary
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def find_footer_region(self, image, footer_height_ratio=0.15):
        """
        Find likely footer region where scale information is typically located.
        
        Args:
            image: Input image
            footer_height_ratio: Ratio of image height to consider as footer
            
        Returns:
            Footer region coordinates (x, y, w, h)
        """
        h, w = image.shape[:2]
        footer_height = int(h * footer_height_ratio)
        
        return (0, h - footer_height, w, footer_height)
    
    def extract_scale_info(self, image, search_regions=None):
        """
        Extract scale/measurement information from SEM image.
        
        Args:
            image: Input SEM image
            search_regions: List of regions to search, or None for auto-detection
            
        Returns:
            Dictionary with extracted scale information
        """
        if search_regions is None:
            # Search in footer region by default
            footer_region = self.find_footer_region(image)
            search_regions = [footer_region]
        
        scale_info = {
            'frame_width_um': None,
            'magnification': None,
            'scale_bar_um': None,
            'extracted_text': [],
            'success': False
        }
        
        for region in search_regions:
            text = self.extract_text_from_image(image, region)
            if text:
                scale_info['extracted_text'].append(text)
                
                # Try to extract frame width
                frame_width = self._extract_frame_width(text)
                if frame_width:
                    scale_info['frame_width_um'] = frame_width
                    scale_info['success'] = True
                
                # Try to extract magnification
                magnification = self._extract_magnification(text)
                if magnification:
                    scale_info['magnification'] = magnification
                
                # Try to extract scale bar
                scale_bar = self._extract_scale_bar(text)
                if scale_bar:
                    scale_info['scale_bar_um'] = scale_bar
        
        return scale_info
    
    def _extract_frame_width(self, text):
        """Extract frame width from text."""
        # Look for patterns like "Frame Width: 21.8 μm"
        patterns = [
            r'[Ff]rame\s*[Ww]idth\s*[:|=]?\s*(\d+(?:\.\d+)?)\s*(?:μm|um|µm)',
            r'[Ww]idth\s*[:|=]?\s*(\d+(?:\.\d+)?)\s*(?:μm|um|µm)',
            r'(\d+(?:\.\d+)?)\s*(?:μm|um|µm)\s*[Ww]idth',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def _extract_magnification(self, text):
        """Extract magnification from text."""
        for pattern in self.magnification_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    mag_str = match.group(1).replace(',', '')
                    return int(mag_str)
                except ValueError:
                    continue
        
        return None
    
    def _extract_scale_bar(self, text):
        """Extract scale bar information from text."""
        for pattern in self.scale_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    # Convert to microns if needed
                    if 'nm' in match.group(0).lower():
                        value = value / 1000.0  # nm to μm
                    elif 'mm' in match.group(0).lower():
                        value = value * 1000.0  # mm to μm
                    return value
                except ValueError:
                    continue
        
        return None
    
    def calculate_pixel_size(self, scale_info, image_width):
        """
        Calculate microns per pixel from scale information.
        
        Args:
            scale_info: Scale information dictionary
            image_width: Width of image in pixels
            
        Returns:
            Microns per pixel, or None if cannot calculate
        """
        if scale_info.get('frame_width_um') and image_width:
            return scale_info['frame_width_um'] / image_width
        
        # Could add other calculation methods here based on scale bar, etc.
        
        return None
    
    def extract_all_metadata(self, image):
        """
        Extract comprehensive metadata from SEM image.
        
        Args:
            image: Input SEM image
            
        Returns:
            Dictionary with all extracted metadata
        """
        # Get basic scale info
        scale_info = self.extract_scale_info(image)
        
        # Calculate pixel size if possible
        h, w = image.shape[:2]
        pixel_size = self.calculate_pixel_size(scale_info, w)
        
        metadata = {
            'image_dimensions': (w, h),
            'frame_width_um': scale_info.get('frame_width_um'),
            'magnification': scale_info.get('magnification'),
            'scale_bar_um': scale_info.get('scale_bar_um'),
            'um_per_pixel': pixel_size,
            'extraction_success': scale_info.get('success', False),
            'extracted_text': scale_info.get('extracted_text', []),
            'ocr_engine': self.ocr_engine
        }
        
        return metadata


def create_ocr_processor(engine='auto', tesseract_path=None):
    """
    Create OCR processor with automatic engine selection.
    
    Args:
        engine: 'auto', 'tesseract', or 'easyocr'
        tesseract_path: Path to tesseract executable
        
    Returns:
        SEMImageOCR instance
    """
    if engine == 'auto':
        if EASYOCR_AVAILABLE:
            engine = 'easyocr'
        elif TESSERACT_AVAILABLE:
            engine = 'tesseract'
        else:
            raise ImportError("No OCR engine available")
    
    return SEMImageOCR(engine, tesseract_path)


def extract_scale_from_image(image_path, ocr_engine='auto'):
    """
    Convenience function to extract scale from image file.
    
    Args:
        image_path: Path to SEM image
        ocr_engine: OCR engine to use
        
    Returns:
        Metadata dictionary
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Create OCR processor
    ocr = create_ocr_processor(ocr_engine)
    
    # Extract metadata
    metadata = ocr.extract_all_metadata(image)
    metadata['image_path'] = image_path
    
    return metadata


# Utility functions for common operations
def get_frame_width_from_footer(image, ocr_engine='auto'):
    """Extract frame width specifically from image footer."""
    ocr = create_ocr_processor(ocr_engine)
    footer_region = ocr.find_footer_region(image)
    text = ocr.extract_text_from_image(image, footer_region)
    return ocr._extract_frame_width(text)


def validate_scale_extraction(metadata, expected_range=(1.0, 1000.0)):
    """
    Validate extracted scale information.
    
    Args:
        metadata: Extracted metadata dictionary
        expected_range: Expected range for frame width in microns
        
    Returns:
        Boolean indicating if extraction seems valid
    """
    frame_width = metadata.get('frame_width_um')
    if frame_width is None:
        return False
    
    return expected_range[0] <= frame_width <= expected_range[1]