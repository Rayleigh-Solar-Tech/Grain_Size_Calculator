#!/usr/bin/env python3
"""
Debug script to test footer detection without GUI
"""

import sys
import os
sys.path.append('src')

from core.exact_footer_ocr import ExactFooterOCR

def test_footer_detection(image_path):
    """Test footer detection on a specific image."""
    print(f"Testing footer detection on: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image file does not exist: {image_path}")
        return
    
    # Create OCR instance
    ocr = ExactFooterOCR()
    
    # Test the analysis
    try:
        print("=" * 50)
        print("TESTING FOOTER DETECTION")
        print("=" * 50)
        
        results = ocr.analyze_sem_footer_exact(image_path)
        
        print("\n" + "=" * 50)
        print("RESULTS:")
        print("=" * 50)
        
        if 'error' in results:
            print(f"❌ ERROR: {results['error']}")
        else:
            print("✅ SUCCESS!")
            print(f"Image size: {results.get('image_size', 'Unknown')}")
            
            metadata = results.get('metadata', {})
            if metadata:
                print("Extracted metadata:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
            else:
                print("No metadata extracted")
            
            if 'ocr_text' in results:
                print(f"\nOCR Text (first 200 chars):")
                print(f"'{results['ocr_text'][:200]}...'")
        
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_footer_detection(image_path)
    else:
        print("Usage: python debug_footer.py <image_path>")
        print("Example: python debug_footer.py 'C:\\path\\to\\your\\sem_image.tiff'")