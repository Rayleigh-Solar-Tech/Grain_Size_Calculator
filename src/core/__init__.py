"""
Core module for Grain Size Calculator.
Contains image processing, SAM analysis, OCR, and results processing components.
"""

__version__ = "1.0.0"
__author__ = "Rayleigh Solar Tech"

# Core module imports
from .config import ConfigManager, AnalysisVariant, ProcessingConfig
from .image_processing import (
    load_and_convert_to_grayscale,
    param_enhance,
    prep_rgb8,
    normalize01,
    create_overlay_visualization
)
# Import SAM analysis lazily to avoid slow PyTorch import on startup
# from .sam_analysis import (
#     GrainAnalyzer,
#     FeretCalculator, 
#     GrainMeasurements,
#     create_complete_analyzer
# )
from .ocr import SEMImageOCR, create_ocr_processor, extract_scale_from_image
from .results import (
    ResultsProcessor,
    ResultsExporter,
    ResultsVisualizer,
    StatisticalSummary,
    create_complete_results_processor
)

__all__ = [
    # Config
    'ConfigManager',
    'AnalysisVariant', 
    'ProcessingConfig',
    
    # Image processing
    'load_and_convert_to_grayscale',
    'param_enhance',
    'prep_rgb8', 
    'normalize01',
    'create_overlay_visualization',
    
    # SAM analysis
    'GrainAnalyzer',
    'FeretCalculator',
    'GrainMeasurements', 
    'create_complete_analyzer',
    
    # OCR
    'SEMImageOCR',
    'create_ocr_processor',
    'extract_scale_from_image',
    
    # Results
    'ResultsProcessor',
    'ResultsExporter', 
    'ResultsVisualizer',
    'StatisticalSummary',
    'create_complete_results_processor'
]

# Lazy import function for SAM analysis (to avoid slow PyTorch startup)
def get_sam_analysis():
    """Lazy import SAM analysis modules only when needed."""
    try:
        from .sam_analysis import (
            GrainAnalyzer,
            FeretCalculator, 
            GrainMeasurements,
            create_complete_analyzer
        )
        return {
            'GrainAnalyzer': GrainAnalyzer,
            'FeretCalculator': FeretCalculator,
            'GrainMeasurements': GrainMeasurements,
            'create_complete_analyzer': create_complete_analyzer
        }
    except ImportError as e:
        print(f"Warning: SAM analysis not available: {e}")
        return None

# Make SAM functions available but don't import them yet
def create_complete_analyzer(*args, **kwargs):
    """Wrapper for lazy loading of SAM analyzer."""
    sam_modules = get_sam_analysis()
    if sam_modules:
        return sam_modules['create_complete_analyzer'](*args, **kwargs)
    else:
        raise ImportError("SAM analysis modules not available")